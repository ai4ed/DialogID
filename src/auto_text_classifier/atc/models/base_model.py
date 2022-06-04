import numpy as np
from atc.utils.data_utils import init_dir, load_df, DataGet
from atc.utils.metrics_utils import get_model_metrics, get_multi_class_report,refit_map

import torch
import random
import os
import pandas as pd
import traceback
from tqdm import tqdm
import time


class BaseModel():
    def __init__(self, config):
        self.model = None
        self.config = config
        self.batch_size = int(self.config.get('batch_size', 32))
        self.max_len = int(self.config.get('max_len', 128))
        self.epochs = int(self.config.get("epochs", 100))
        self.patience = int(self.config.get("patience", 5))
        #
        self.save_dir = self.config.get('save_dir', "")
        self.train_dir = self.config.get('train_dir', "")
        self.dev_dir = self.config.get('dev_dir', "")
        self.test_dir = self.config.get('test_dir', "")
        #
        self.model_dir = self.config.get('model_dir', "")
        self.num_labels = int(self.config.get('num_labels', 2))
        self.seed = int(self.config.get('seed', 0))
        self.fp16 = self.config.get('fp16', None)
        self.token_type_ids_disable = self.config.get(
            'token_type_ids_disable', False)
        if self.num_labels == 2:
            refit = self.config.get('refit', 'acc')  # support
            self.refit = refit_map[refit]
        else:
            self.refit = refit_map['acc']
        self.adt_type = self.config.get('adt_type',None)  # adversarial_training
        self.focal_loss = self.config.get('focal_loss', 0)
        self.supcon_loss = self.config.get('supcon_loss', 0)
        self.triplet_loss = self.config.get('triplet_loss', 0)
        self.K = self.config.get('K', 3)
        self.fgm_epsilon = self.config.get('fgm_epsilon', 3.5e-5)
        self.lr = self.config.get('lr',2e-5)
        self.eval_steps = self.config.get("eval_steps", None)
        self.multi_label = self.config.get('multi_label', False)
        #
        self.date = time.strftime("%Y-%m-%d", time.localtime())
        #
        self.pos_weight = self.config.get('pos_weight', False)
        #
        # 是否使用模型最顶层token向量的平均embedding替换cls作为
        self.mean_top_level_embedding = self.config.get(
            'mean_top_level_embedding', False)
        #
        # 是否使用模型最顶层与label文本进行attention
        self.top_level_embedding_attention_with_label = self.config.get(
            'top_level_embedding_attention_with_label', False)
        #
        init_dir(self.save_dir)

    def train(self):
        """train model use train_path
        Parameters
        ----------
            model_path: model_path
        Returns
        -------
            report:model performance in test
        """
        raise NotImplementedError

    def load_model(self, model_path):
        """load model from model_path
        Parameters
        ----------
            model_path: model_path
        Returns
        -------
            None
        """
        raise NotImplementedError

    def demo(self, text):
        """demo for one text
        Parameters
        ----------
            text: input text
        Returns
        -------
            p:the probability of text
        """
        raise NotImplementedError

    def demo_text_list(self, text_list):
        """demo input text_list
        Parameters
        ----------
            text_list: text_list
        Returns
        -------
            p_list:the probability of all text
        """
        raise NotImplementedError

    def predict(self, text):
        """
        text: str
        """
        return self.demo(text)

    def predict_list(self, text_list):
        return self.demo_text_list(text_list)

    def evaluate(self, df, single_sample=False):
        df = load_df(df)
        y_pred = []
        if single_sample:
            for text in tqdm(df['text'].tolist()):
                y_pred.append(self.demo(text))
        else:
            y_pred = self.demo_text_list(df['text'].tolist())
        #
        if self.multi_label:
            y_pred = np.array(y_pred)
            y_true = [eval(x) for x in df['label'].tolist()]
        else:
            y_pred = np.array(y_pred)
            y_true = np.array(df['label'])
        #
        if self.num_labels == 2:
            report = get_model_metrics(y_true, y_pred)
        else:
            report = get_multi_class_report(y_true, y_pred)
        return report

    def release(self):
        pass

    def set_seed(self, seed=-1):
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def train_cv(self, df, cv):
        df = load_df(df)
        data_get = DataGet(df=df,n_splits=cv,random_state=self.seed)
        root_dir = self.save_dir
        report_list = []
        try:
            for kf_i in range(cv):
                print("Start cv {}/{}".format(kf_i+1,cv))
                self.save_dir = os.path.join(root_dir, str(kf_i))
                df_train, df_dev, df_test = data_get.get_data(kf_i=kf_i)
                report = self.train(df_train, df_dev, df_test)
                report['kf_i'] = kf_i
                report_list.append(report)
                print("Finish cv {}/{}".format(kf_i+1,cv))
                self.release()
        except Exception as e:
            print(traceback.format_exc())
        finally:
            self.save_dir = root_dir  # 避免修改全局变量
        return pd.DataFrame(report_list) 

    def eval_cv(self, df, cv):
        df = load_df(df)
        root_dir = self.save_dir
        try:
            kf_name_list = []
            for kf_i in range(cv):
                print("Start cv {}/{}".format(kf_i+1,cv))
                model_dir = os.path.join(root_dir, str(kf_i))
                _ = self.load_model(model_dir)
                kf_name = 'kf_{}'.format(kf_i)
                kf_name_list.append(kf_name)
                df[kf_name] = self.predict_list(df['text'].tolist())
                self.release()
                print("Finish cv {}/{}".format(kf_i+1,cv))
            df['kf_avg'] = df[kf_name_list].mean(axis=1)
        except Exception as e:
            print(traceback.format_exc())
        finally:
            self.save_dir = root_dir  # 避免修改全局变量
        return df
