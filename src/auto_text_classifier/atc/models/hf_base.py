
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
import numpy as np
import pandas as pd
import random
import datetime
from tqdm import tqdm, trange
from transformers import BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.data.data_collator import default_data_collator
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from atc.utils.data_utils import init_dir
from atc.models.base_model import BaseModel
from atc.utils.metrics_utils import get_model_metrics
from atc.utils.data_utils import load_df, load_df_1
from transformers import AutoConfig
from atc.utils.adt_utils import *
from atc.utils.data_utils import DFDataset

import gc
import time
import sys

try:
    from apex import amp  # noqa: F401
    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


def get_model_report(preds, labels, num_labels, multi_label=False):
    # 多标签
    if multi_label:
        pred_list_01 = HFBase.transfer_01(preds)
        correct_num = 0
        for i in range(len(pred_list_01)):
            if sum(pred_list_01[i] == labels[i]) == num_labels:
                correct_num += 1
        acc = correct_num / len(labels)
        return {"Accuracy": acc}
    #
    if num_labels != 2:
        # 多分类
        pred_flat = np.argmax(preds, axis=1)
        acc = np.sum(pred_flat == labels) / len(labels)
        return {"Accuracy": acc}
    else:
        # 二分类
        y_pred = preds[:, 1]
        return get_model_metrics(y_true=labels, y_pred=y_pred)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class HFBase(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.tokenizer = self.get_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = self.save_dir
        self.config = config
        self.adt_emb_name = config.get("adt_emb_name","emb")
        self.adt_epsilon = config.get("adt_epsilon",1)
        #
        if self.pos_weight:
            self.pos_counts_dict = self.get_pos_count()  # 训练集中各个类别的样本数
            self.pos_weight = self.get_pos_weight(self.pos_counts_dict)      # loss中各个类别的权重
        else:
            self.pos_weight = None
        #
        if self.focal_loss == 1:
            self._loss_fun = FocalLoss(logits=True, multilabel=self.multi_label)
            print("Training use focal loss ~~")
        elif self.supcon_loss == 1:
            self._loss_fun = SupConLoss(config["num_labels"])
            print("Training use supcon loss ~~")
        elif self.triplet_loss == 1:
            self._loss_fun = TripletLoss()
            print("Training use triplet loss ~~")
        elif self.multi_label:
            self._loss_fun = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=self.pos_weight)
            print("Training use BCEWithLogitsLoss ~~, weight {}".format(self.pos_weight))
        else:
            self._loss_fun = nn.CrossEntropyLoss(weight=self.pos_weight)  # 默认损失函数未交叉熵损失，不配置权重
            print("Training use origin loss ~~, weight {}".format(self.pos_weight))
        #
        self.model_to_save = None

    def get_pos_count(self):
        """
        计算训练集中每个类别的数量
        """
        df_tmp = pd.read_csv(self.train_dir)
        label_count_dict = dict()  # {label: count}
        if "label_index" in df_tmp.columns.tolist():
            multilabel_list = df_tmp["label_index"].tolist()
            for label_list in multilabel_list:
                label_list = eval(label_list)
                for label in label_list:
                    if label in label_count_dict:
                        label_count_dict[label] += 1
                    else:
                        label_count_dict[label] = 1
        else:
            label_list = df_tmp["label"].tolist()
            for label in label_list:
                if label in label_count_dict:
                    label_count_dict[label] += 1
                else:
                    label_count_dict[label] = 1
        #
        return label_count_dict

    def get_pos_weight(self, pos_counts_dict):
        """
        计算loss中各个类别的权重
        weight[i] = min(counts) / counts[i] 最少样本的类别权重为1，其余类别样本越多权重越低
        """
        pos_counts_list = [0] * len(pos_counts_dict)
        for index, count in pos_counts_dict.items():
            pos_counts_list[index] = count
        pos_weight = [max(pos_counts_list) / count for count in pos_counts_list]
        #
        return torch.Tensor(pos_weight).to(self.device)

    def get_tokenizer(self):
        raise NotImplementedError

    def get_data_generator(self, data, shuffle=False, num_workers=1):
        data = load_df_1(data)
        dataset = DFDataset(data,
                            tokenizer=self.tokenizer,
                            max_len=self.max_len,
                            multi_label=self.multi_label,
                            num_labels=self.num_labels)
        data_dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=default_data_collator,
            batch_size=self.batch_size,
        )
        return data_dataloader

    def get_inputs(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        return batch

    def process_data(self, train_path, dev_path, test_path):
        train_generator = self.get_data_generator(train_path, shuffle=True)
        dev_generator = self.get_data_generator(dev_path)
        test_generator = self.get_data_generator(test_path)
        return train_generator, dev_generator, test_generator

    def init_model(self):
        print("HFBase init")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir,
                                                                            num_labels=self.num_labels)
                                                                            #output_hidden_states=True)
        except:
            config = self.get_config()
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir,
                                                                            config=config)
                                                                            #output_hidden_states=True)

    def train(self, train_path, dev_path, test_path):
        self.set_seed(self.seed)  # 为了可复现
        train_generator, dev_generator, test_generator = self.process_data(
            train_path, dev_path, test_path)
        self.init_model()
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=self.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                               # args.adam_epsilon  - default is 1e-8.
                               eps=1e-8
                               )
        if not self.fp16 is None:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16)
            print("train model use fp16")
        self.train_model(train_generator,
                           dev_generator, test_generator)
        ## load best model
        self.load_model(self.save_dir)
        return self.evaluate(test_path)

    def get_sentence_embedding(self, text):
        """
        使用当前模型预测text，获得embedding
        """
        pass

    def get_label_attention_sentence_embedding(self):
        """
        使用模型同时对sentence和label进行预测，求word的attention再合并成sentence embedding
        https://arxiv.org/pdf/1805.04174.pdf
        """
        pass

    def load_model(self, model_path):
        # 有一些模型必须要指定num_labels,例如bart,但是有一些模型有没有这个参数，因此这里首先判断。和init_model很类似
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                            num_labels=self.num_labels)
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Copy the model to the GPU.
        self.model = self.model.to(self.device)

    def _eval_model(self, dataloader, have_label=False):
        self.model.eval()
        total_loss = 0
        pred_list = []
        label_list = []
        # Predict
        batch_num = 0
        for batch in tqdm(dataloader):
            batch_num += 1
            inputs = self.get_inputs(batch)
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(**inputs)
                total_loss += outputs[0].item()
                if self.multi_label:
                    pred = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
                else:
                    pred = F.softmax(outputs['logits']).detach().cpu().numpy()
                pred_list.append(pred)
                label_list.append(batch['labels'].detach().cpu().numpy())
        #
        y_pred = np.concatenate(pred_list)
        labels = np.concatenate(label_list)
        #
        if have_label:
            loss = total_loss/batch_num
        else:
            loss, labels = None, None
        #
        return loss, y_pred, labels

    def demo(self, text, softmax_b=False):
        """
        对单条数据进行预测
        """
        if text.count("[SEP]") == 1:
            text1, text2 = text.split("[SEP]")
        else:
            text1 = text
            text2 = None
        #
        encoding = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        for k, v in encoding.items():
            v = torch.Tensor([v]).long()
            encoding[k] = v.to(self.device)
        #
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
        if self.multi_label:
            preds = torch.sigmoid(outputs['logits']).detach().cpu().numpy()
        else:
            preds = F.softmax(outputs['logits']).detach().cpu().numpy()
        #
        if self.config.get("后处理", False):
            preds = self.post_thresholds(preds)
        #
        if softmax_b:
            return preds
        #
        pred = []
        if self.multi_label:
            for p in preds.tolist()[0]:
                if p >= 0.5:
                    pred.append(1)
                else:
                    pred.append(0)
        else:
            pred = preds.argmax()
        return pred

    @staticmethod
    def transfer_01(preds, threshold=0.5):
        pred_list = []
        if type(preds) != list:
            preds = preds.tolist()
        #
        for p in preds:
            pred = []
            for val in p:
                if val >= threshold:
                    pred.append(1)
                else:
                    pred.append(0)
            #
            pred_list.append(np.array(pred))
        #
        return np.array(pred_list)

    def demo_text_list(self, text_list, softmax_b=False):
        """
        对多条数据进行预测
        """
        df = pd.DataFrame({"text": text_list})
        dataloader = self.get_data_generator(df, shuffle=False)
        _,preds,_ = self._eval_model(dataloader, have_label=False)
        #
        # df1 = pd.DataFrame()
        # df1["softmax"] = preds.tolist()
        model_name = self.model_dir.split("/")[-2]
        # df1.to_csv("/data1/sp/jupyter_data/exercise/output/softmax_result_{}_{}.csv".format(model_name, self.date))
        #
        if self.config.get("后处理", False):
            preds = self.post_thresholds(preds)
        #
        if softmax_b:
            return preds
        #
        pred_list = []
        if self.multi_label:
            # 多标签
            pred_list = self.transfer_01(preds)
        else:
            if self.num_labels == 2:
                # 二分类
                pred_list = preds[:, 1]
            else:
                # 多分类
                pred_list = np.argmax(preds, axis=1).flatten()
        #
        return pred_list

    def post_thresholds(self, input_preds):
        # 对每个位置，如果这个地方的数字没有大于阈值，则最后一位+1
        # 如果大于阈值则还是原来的样子.

        #             鼓励，引导， 总结， 寒暄， 笔记，复述，复习，举例，其他
        # thresholds = [0] * 9
        thresholds = [0.98,  0.98,  0.99,   0.9,   0.95, 0.99,  0.99,   0.99, 0]

        preds = input_preds.copy()
        origin_pred_class = np.argmax(input_preds, axis=1).flatten()

        for idx, i in enumerate(origin_pred_class):
            if input_preds[idx, i] < thresholds[i]:
                preds[idx, 8] += 0.9
        return preds

    def labelEncoder(self, y, nClass):
        """
        将label转换成one hot矩阵
        [3, 4, 1] -> [[0,0,0,1,0], [0,0,0,0,1], [0,1,0,0,0]]
        """
        tmp = torch.zeros(size=(y.shape[0], nClass))
        for i in range(y.shape[0]):
            tmp[i][y[i]] = 1
        return tmp.to(self.device)

    def k_is_in(self, t, keywords):
        for k in keywords:
            if k in t:
                return True
        return False

    def train_model(self, train_generator, dev_generator, test_generator):
        patience_count = 0
        best_eval_score = 0
        best_loss = np.inf
        epochs = self.epochs
        output_dir = self.save_dir
        total_steps = len(train_generator) * epochs

        # Create the learning rate scheduler.
        # It is useful to release gpu momory.
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        # if self.adt_type=='fgm':
        #     fgm = FGM(self.model)
        #     print("Training use fgm ~~, fgm_epsilon is {}".format(self.fgm_epsilon))

        if self.adt_type=='fgm':
            fgm = FGM(self.model)
            print(f"Training use FGM ~~,self.adt_epsilon is {self.adt_epsilon}")
        elif self.adt_type == 'pgd':
            pgd = PGD(self.model)
            print("Training use PGD ~~")
        elif self.adt_type == 'freeat':
            freeat = FreeAT(self.model)
            print("Training use FreeAT ~~")
        elif self.adt_type == 'freelb':
            freelb = FreeLB(self.model)
            print("Training use FreeLB ~~")
        else:
            print("Training use none adt ~~")

        # Store the average loss after each epoch so we can plot them.
        loss_values = []
        # For each epoch...
        self.set_seed(self.seed)  # 为了可复现
        step_num = 0

        if(self.adt_type == "freeat"):
            self.epochs = int(self.epochs / self.K)

        for epoch_i in range(0, epochs):
            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            print("learning rate: {}".format(self.lr))

            # Measure how long the training epoch takes.
            t_train = time.time()
            # Reset the total loss for this epoch.
            total_loss = 0
            # For each batch of training data...
            batch_i = 0
            for _, batch in tqdm(enumerate(train_generator)):
                batch_i += 1
                self.model.train()
                step_num += 1
                inputs = self.get_inputs(batch)
                outputs = self.model(**inputs)
                #
                logit = outputs[1]
                #
                loss = self._loss_fun(logit, inputs['labels'])
                total_loss += loss
                loss.backward()  # 反向传播，得到正常的grad
                if batch_i % 100 == 0:
                    print("batch {} loss {} \n".format(batch_i, loss))
                    sys.stdout.flush()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # if self.adt_type=='fgm':
                #     # 对抗训练
                #     fgm.attack(epsilon=self.fgm_epsilon) # 在embedding上添加对抗扰动
                #     outputs = self.model(**inputs)
                #     loss_adv = outputs[0]
                #     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                #     fgm.restore() # 恢复embedding参数

                if self.adt_type=='fgm':#使用fgm对抗训练方式
                    #对抗训练
                    fgm.attack(self.adt_epsilon, self.adt_emb_name) # 在embedding上添加对抗扰动
                    outputs = self.model(**inputs)
                    loss_adv = outputs[0]
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore(self.adt_emb_name)  # 恢复embedding参数
                    #梯度下降更新参数
                    self.optimizer.step()
                elif self.adt_type == 'pgd':#使用pgd对抗训练方式
                    pgd.backup_grad()
                    # 对抗训练
                    for t in range(self.K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != self.K - 1:
                            self.optimizer.zero_grad()
                        else:
                            pgd.restore_grad()
                        outputs = self.model(**inputs)
                        loss_adv = outputs[0]
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()  # 恢复embedding参数
                    self.optimizer.step()
                elif self.adt_type == 'freeat':  # 使用freeat对抗训练方式
                    # 对抗训练
                    for t in range(self.K):
                        freeat.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        self.optimizer.zero_grad()
                    
                        # freeat.restore_grad()
                        outputs = self.model(**inputs)
                        loss_adv = outputs[0]
                        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        self.optimizer.step()
                    # freeat.restore() # 恢复embedding参数
                elif self.adt_type == 'freelb':  # 使用freelb对抗训练方式
                    freelb.backup_grad()
                    # 对抗训练
                    for t in range(self.K):
                        freelb.attack(is_first_attack=(t==0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                        # self._optimizer.zero_grad()
                        outputs = self.model(**inputs)
                        loss_adv = outputs[0] / self.K
                        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    freelb.restore()  # 恢复embedding参数
                    self.optimizer.step()
                else:
                    self.optimizer.step()

                # 梯度下降，更新参数
                # self.optimizer.step()
                # Update the learning rate.
                scheduler.step()
                self.model.zero_grad()
                # self.eval_steps 个step进行一次效果评估（此处未执行）
                if self.eval_steps is not None and self.eval_steps == step_num:
                    t0 = time.time()
                    avg_eval_loss, y_pred, labels = self._eval_model(dev_generator, have_label=True)
                    model_report = get_model_report(y_pred, labels, self.num_labels, self.multi_label)
                    eval_score = model_report[self.refit]  # 选取优化的指标
                    # if best save self.model
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        print(" Get best result, saving self.model to %s" % output_dir)
                        self.model_to_save = self.model.module if hasattr(
                            self.model, 'module') else self.model
                        self.model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                    # Report the final accuracy for this validation run.
                    print("  Validation {}: {:.4f},Loss :{:.4f},best_eval_loss {} is {:.4f}".format(self.refit,
                                                                                eval_score,
                                                                                avg_eval_loss,
                                                                                self.refit,
                                                                                best_eval_score))
                    print("  Validation took: {:}".format(
                        format_time(time.time() - t0)))

                    step_num = 0  # reset step_num
            #
            # 每个epoch进行一次效果评估
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_generator)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                format_time(time.time() - t_train)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

             # do eval
            # Put the self.model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            
            avg_eval_loss, y_pred, labels = self._eval_model(dev_generator, have_label=True)
            model_report = get_model_report(y_pred, labels, self.num_labels, self.multi_label)
            eval_score = model_report[self.refit]  # 选取优化的指标
            
            # Report the final accuracy for this validation run.
            print("  {}: {:.4f},Loss :{:.4f}".format(self.refit,eval_score,avg_eval_loss))
            print("  Validation took: {:}".format(
                format_time(time.time() - t0)))
            
            # if best save self.model
            if eval_score > best_eval_score + 0.001:
                patience_count = 0
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print("Get best result, saving self.model to %s" % output_dir)
                self.model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                self.model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                best_eval_score = eval_score
            else:
                patience_count = patience_count + 1
            if patience_count > self.patience:
                print("Epoch {}:early stopping Get best result, {} did not improve from {}".format(
                    epoch_i + 1,self.refit,best_eval_score))
                break
            # 学习率衰减
            self.lr *= 0.9
        #
        del self.optimizer
        del self.model_to_save
        if self.adt_type == 'fgm':
            del fgm

    def release(self):
        # see this issue:https://github.com/huggingface/transformers/issues/1742
        print("Release model")
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_raw_config(self):
        '''获取原始的config'''
        config = AutoConfig.from_pretrained(self.model_dir)
        return config

    def get_config(self):
        config = self.load_raw_config()
        num_labels = self.num_labels
        config_dict = {"num_labels": num_labels,
                    "id2label": {x: "LABEL_{}".format(x) for x in range(num_labels)},
                    "label2id": {"LABEL_{}".format(x): x for x in range(num_labels)},
                    "output_hidden_states": self.config["output_hidden_states"],
                    "label_text_filepath": self.config["label_text_filepath"],
                    "max_length": self.max_len,
                    "model_dir": self.model_dir,
                    }
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config