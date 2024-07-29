import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer


# last_n_layers = 4
# sent_emb_mode = "cls"

class BertEncoder(nn.Module):
    def __init__(
            self, 
            bert_type="./../Bio_ClinicalBERT", 
            force_download=False,
            last_n_layers=4,
            aggregate_method="sum",
            norm=False,
            embedding_dim=768,
            freeze_bert=False,
            agg_tokens=True,
            agg_sents=True,
            sent_emb_mode="cls",
        ):
        super(BertEncoder, self).__init__()

        # self.bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        self.bert_type = bert_type
        self.last_n_layers = last_n_layers
        self.aggregate_method = aggregate_method
        self.norm = norm
        self.embedding_dim = embedding_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = agg_tokens
        self.agg_sents = agg_sents
        self.sent_emb_mode = sent_emb_mode.lower()

        print(f"Loading bert model {self.bert_type} (force_download: {force_download})...")
        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True, force_download=force_download
        ).cuda()

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)  # batch, num_words, layers, dim
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def generate_segmask_from_sents(self, sents):
        sents = np.array(sents)
        segments = (sents == ".").cumsum(axis=1)
        segments[sents == "."] = -1
        segments[sents == "[PAD]"] = -1
        segments[sents == "[SEP]"] = -1
        segments[:,0] = -1  # [CLS] token
        segments = torch.from_numpy(segments).to(self.model.device)
        return segments

    def aggregate_segments(self, segments, embeddings):
        """
        Aggregate embeddings based on segments

        segments: batch_size, num_words
        embeddings: batch_size, num_words, embedding_dim
        """
        # embeddings = embeddings[:,1:,:]

        max_seg_num = segments.max()+1
        batch_size, num_token, embedding_dim = embeddings.shape
        seg_embeddings = torch.zeros(batch_size, max_seg_num, embedding_dim).to(embeddings.device)
        seg_mask = torch.ones(batch_size, max_seg_num, dtype=torch.bool).to(embeddings.device)
        
        for b in range(batch_size):
            seg_num = segments[b].max()+1
            seg_mask[b, :seg_num] = False
            for seg in range(seg_num):
                seg_embeddings[b, seg] = embeddings[b, segments[b]==seg].sum(axis=0)

        return seg_embeddings, seg_mask

    def forward(self, ids, attn_mask, token_type):

        outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

        if self.sent_emb_mode == "cls":
            sent_embeddings = outputs[0][:, 0, :].contiguous()

        # if segments is not None:
        if self.agg_sents:
            segments = self.generate_segmask_from_sents(sents)
            seg_embeddings, seg_mask = self.aggregate_segments(segments, word_embeddings)

        # batch_dim, num_words, feat_dim = word_embeddings.shape
        # word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        # if self.emb_local is not None:
        #     word_embeddings = self.emb_local(word_embeddings)
        # word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        # word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        if self.agg_sents:
            return word_embeddings, sent_embeddings, sents, seg_embeddings, seg_mask
        return word_embeddings, sent_embeddings, sents

    def encode(self, text, max_length=100):
        if type(text) == str:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            return encoded.input_ids, encoded.attention_mask, encoded.token_type_ids
        elif type(text) == list:
            input_ids, attn_mask, type_ids = [], [], []
            for t in text:
                encoded = self.tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
                input_ids.append(encoded.input_ids)
                attn_mask.append(encoded.attention_mask)
                type_ids.append(encoded.token_type_ids)
            return (
                torch.stack(input_ids),
                torch.stack(attn_mask),
                torch.stack(type_ids),
            )
            