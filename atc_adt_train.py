import sys
sys.path.append('src/auto_text_classifier')


import os
from atc.models.aml import AML

# 3、选择数据
train_path = "data/train.csv"
dev_path = "data/dev.csv"
test_path = "data/test.csv"

# 4、训练模型
config = dict()
config['num_labels'] = 9
config['epochs'] = 100
config['batch_size'] = 64
config['max_len'] = 128
config['lr'] = 0.00001
config['adt_type'] = "fgm"
config['adt_emb_name'] = 'emb'

for adt_eps in [0.5,1]:
    config['adt_epsilon'] = adt_eps
    model_list = ['roberta']
    save_dir = f"output/fgm/adt_eps={adt_eps}"
    model = AML(save_dir=save_dir, config=config)
    df_report = model.train(train_path, dev_path, test_path, model_list=model_list) 
    save_name = "-".join(model_list)
    df_report.to_csv(os.path.join(save_dir,f'result_{save_name}.csv'), index=True)