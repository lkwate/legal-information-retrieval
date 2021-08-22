import torch
from transformers import AutoModel, AutoTokenizer
from pyserini.dindex import DocumentEncoder
import faiss


class LongAutoDocumentEncoder(DocumentEncoder):
    def __init__(
        self,
        model_name,
        tokenizer_name=None,
        device="cuda:0",
        pooling="cls",
        max_length=2048,
        l2_norm=False,
    ):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.has_model = True
        self.pooling = pooling
        self.max_length = max_length
        self.l2_norm = l2_norm

    def encode(self, texts, titles=None):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)
        if self.pooling == "mean":
            embeddings = (
                mean_pooling(outputs[0], inputs["attention_mask"])
                .detach()
                .cpu()
                .numpy()
            )
        else:
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            faiss.normalize_L2(embeddings)
        return embeddings


def mean_pooling(last_hidden_state, attention_mask):
    token_embeddings = last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
