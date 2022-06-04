from atc.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification
from transformers import AdamW

class ELECTRA(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'electra'

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return tokenizer