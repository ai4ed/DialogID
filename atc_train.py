import sys
sys.path.append('src/auto_text_classifier')
import os
from atc.models.aml import AML

train_path = "data/train.csv"
dev_path = "data/dev.csv"
test_path = "data/test.csv"


config = dict()
config['num_labels'] = 9
config['epochs'] = 100
config['batch_size'] = 64
config['max_len'] = 128


model_list = ['electra_base','xlnet_base','bert_base','macbert_base']


save_dir = "output/raw"
model = AML(save_dir=save_dir, config=config)
df_report = model.train(train_path, dev_path, test_path, model_list=model_list)  # model_list 可以参考“支持的模型”，不填则使用全部的模型。

df_report.to_csv(os.path.join(save_dir,'result_{}.csv'.format("-".join(model_list))), index=True)
