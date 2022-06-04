from atc.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification
from transformers import AdamW
from transformers import BertConfig

class ROBERTA(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'roberta'
        
    def get_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        except:
            tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        return tokenizer

    def load_model(self, model_path):
        if self.config.get('use_bert_type'):
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Copy the model to the GPU.
        self.model = self.model.to(self.device)
        return self.model

    
    # def load_raw_config(self):
    #     '''获取原始的config'''
    #     config = BertConfig.from_pretrained(self.model_dir)
    #     return config
