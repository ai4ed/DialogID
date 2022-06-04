import os
base_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join
# chinese models

# xlnet base
xlnet_base_dir = join(base_path, '../data/hfl_chinese_xlnet_base')
xlnet_base_config = {"model_dir": xlnet_base_dir,
                     "save_dir": 'model/xlnet_base'}

# bert base
bert_base_dir = join(base_path, '../data/bert_base_chinese')
bert_base_config = {"model_dir": bert_base_dir, 
                    "save_dir": 'model/bert_base',
                    "epochs": 100,
                    }

# chinese-roberta-wwm-ext
chinese_roberta_wwm_ext_dir = join(
    base_path, '../data/chinese_roberta_wwm_ext')
chinese_roberta_wwm_ext_config = {"model_dir": chinese_roberta_wwm_ext_dir,
                                  "save_dir": 'model/chinese_roberta_wwm_ext/'}


# chinese_electra_base
hfl_chinese_electra_base_dir = join(
    base_path, '../data/hfl_chinese_electra_base_d')
hfl_chinese_electra_base_config = {"model_dir": hfl_chinese_electra_base_dir,
                                   "save_dir": 'model/electra_base/'}


# macbert model
## macbert_base
macbert_base_config = {"model_dir":join(base_path, '../data/hfl_chinese_macbert_base'),
                        "save_dir":"model/macbert_base"}