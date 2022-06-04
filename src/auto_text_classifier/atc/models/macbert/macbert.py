from atc.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,AutoTokenizer,AutoModelForSequenceClassification
from transformers import AdamW
from transformers import BertConfig

class MacBERT(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'macbert'
        
    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        return tokenizer

    def load_model(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        # Copy the model to the GPU.
        self.model = self.model.to(self.device)
        return self.model