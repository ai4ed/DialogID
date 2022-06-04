from atc.models import *
from atc.configs import *

model_dict = {
    "macbert_base": {"model_class": MacBERT, "config": macbert_base_config},
    "bert_base": {"model_class": BERT, "config": bert_base_config},
    "roberta": {"model_class": ROBERTA, "config": chinese_roberta_wwm_ext_config},
    "electra_base": {"model_class": ELECTRA, "config": hfl_chinese_electra_base_config},
    "xlnet_base": {"model_class": XLNet, "config": xlnet_base_config},
}




default_model_list = list(model_dict.keys())
