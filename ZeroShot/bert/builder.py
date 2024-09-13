from .bert_encoder import BertEncoder as BERT
from .custom_bert_encoder import BertEncoder as CustomBERT


def build_bert(bert_type="maco", force_download=False):
    if bert_type.lower() == "maco":
        return BERT()
    else:
        print("building bert model...")
        return CustomBERT(bert_type=bert_type, force_download=force_download)

