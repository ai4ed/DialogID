import os
import copy
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.layers import Lambda, Dense
from atc.utils.data_utils import init_dir
from atc.models.base_model import BaseModel
from atc.utils.metrics_utils import get_model_metrics,get_multi_class_report
from atc.utils.data_utils import load_df
from atc.configs.aml_config import model_dict, default_model_list
from atc.utils.data_utils import load_df
import traceback
import json


class AML():
    def __init__(self, save_dir, config={}):
        self.model_dict = model_dict
        self.save_dir = save_dir
        self.config = config
        self.batch_size = int(self.config.get('batch_size', 32))
        self.max_len = int(self.config.get('max_len', 128))
        self.epochs = int(self.config.get("epochs",100))
        self.patience = int(self.config.get("patience", 5))
        self.num_labels = int(self.config.get('num_labels',2))
        init_dir(self.save_dir)

    def get_model_config(self, model_name):
        model_class = copy.deepcopy(self.model_dict[model_name]['model_class'])
        config = copy.deepcopy(self.model_dict[model_name]['config'])
        return model_class, config

    def __evaluate_one_model(self, model, df, model_name, data_set):
        df = load_df(df)
        # add time
        tic = time.time()
        y_pred = model.demo_text_list(df['text'].tolist())
        toc = time.time()
        # cal avg time
        avg_time_s = (toc-tic)/df.shape[0]
        # get report
        #
        if model.multi_label:
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
        report['model_name'] = model_name
        report['data_set'] = data_set
        report['avg_time_s'] = avg_time_s
        return report

    def __check_model_list(self, model_list):
        if len(model_list) == 0:
            return default_model_list

        for model_name in model_list:
            if model_name not in self.model_dict:
                raise Exception(
                    "model:{} is not support now!".format(model_name))
        return model_list

    def __get_one_model(self, model_name, df_train, df_dev, df_test, train=True):
        model_class, config = self.get_model_config(model_name)
        config.update(self.config)
        config['save_dir'] = os.path.join(self.save_dir, model_name)
        print("config is :{}".format(config))
        model = model_class(config)
        if train:
            print('Training...')
            print("Start train {}".format(model_name))
            _ = model.train(df_train, df_dev, df_test)
            print("release after train")
        else:
            print("Load model")
            model.load_model(model.model_path)
            print("Load finish")
        return model

    def __get_report(self, train_path, dev_path, test_path, model_list=[], train=True):
        model_list = self.__check_model_list(model_list)
        # load data
        df_train = load_df(train_path)
        df_dev = load_df(dev_path)
        df_test = load_df(test_path)
        # train or eval all model
        self.all_report = []
        for model_name in tqdm(model_list):
            try:
                # get model
                model = self.__get_one_model(
                    model_name, df_train, df_dev, df_test, train=train)
                # get dev/test report
                dev_report = self.__evaluate_one_model(
                    model, df_dev, model_name, "dev")
                test_report = self.__evaluate_one_model(
                    model, df_test, model_name, "test")
                # append report to list
                self.all_report.append(dev_report)
                self.all_report.append(test_report)
                # release
                model.release()
                print("model_name:{} eval finish!,dev_report:{},test_report:{}".format(
                    model_name, dev_report, test_report))
            except:
                print("model_name:{},fail,detail is {}".format(model_name,traceback.format_exc()))
        if self.num_labels == 2:
            df_report = pd.DataFrame(self.all_report)
            cols = ["Accuracy", "Precision", "Recall",
                    "F_meansure", "AUC_Value", "avg_time_s"]
            df_report_table = df_report.pivot_table(
                index=["data_set", "model_name"], values=cols)[cols]
        else:
            df_report_table = pd.concat(self.all_report)
        return df_report_table

    def fit(self, train_path, dev_path, test_path, model_list=[]):
        """等价于train()
        """        

        df_report = self.__get_report(
            train_path, dev_path, test_path, model_list=model_list, train=True)
        return df_report

    def train(self, train_path, dev_path, test_path, model_list=[]):
        '''等价于fit()'''
        return self.fit(train_path, dev_path, test_path, model_list=model_list)

    def evaluate(self, df_path, model_list):
        '''在df_path上使用model_list进行评估，返回结果。'''
        model_list = self.__check_model_list(model_list)
        train = False
        df = load_df(df_path)
        all_report = []
        for model_name in tqdm(model_list):
            try:
                # get model
                model = self.__get_one_model(
                    model_name, df_train=None, df_dev=None, df_test=None, train=train)
                model_report = self.__evaluate_one_model(model, df, model_name, "")
                all_report.append(model_report)
                # release
                model.release()
                print("model_name:{} eval finish!,model_report:{}".format(
                    model_name, model_report))
            except:
               print("model_name:{},fail,detail is {}".format(model_name,traceback.format_exc()))
        if self.num_labels==2:
            cols = ["model_name", "Accuracy", "Precision",
                    "Recall", "F_meansure", "AUC_Value", "avg_time_s"]
            df_report = pd.DataFrame(all_report)[cols]
        else:
            df_report = pd.concat(all_report)
        return df_report

    def get_list_result(self, df_list, model_list):
        '''获取所有模型的输出结果'''
        model_list = self.__check_model_list(model_list)
        train = False
        df_list = [load_df(x) for x in df_list]
        for model_name in tqdm(model_list):
            # get model
            try:
                model = self.__get_one_model(
                    model_name, df_train=None, df_dev=None, df_test=None, train=train)
                for df in df_list:
                    df[model_name] = model.predict_list(df['text'].tolist())
                # release
                model.release()
            except:
               print("model_name:{},fail,detail is {}".format(model_name,traceback.format_exc()))
        return df_list
    
    def pred_model_list(self, df_list, model_list):
        '''输入df list和model list,返回每个模型在每个df中的预测结果'''
        return self.get_list_result(df_list,model_list)