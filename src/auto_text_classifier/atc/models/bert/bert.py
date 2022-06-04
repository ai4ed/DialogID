from atc.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification
from transformers import AdamW
from transformers import BertConfig

class BERT(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'bert'

    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        return tokenizer

    def load_raw_config(self):
        '''获取原始的config'''
        config = BertConfig.from_pretrained(self.model_dir)
        return config