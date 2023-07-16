import pickle
from itertools import islice
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AlbertModel, AlbertTokenizer


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class DgaDetector:
    def __init__(
        self, pkl_model_path, torch_device="cpu", encoder_model="albert-base-v2"
    ) -> None:
        self.device = torch_device
        self.encoder_model_name = encoder_model
        self.tokenizer = AlbertTokenizer.from_pretrained(self.encoder_model_name)
        self.model = AlbertModel.from_pretrained(self.encoder_model_name)
        self.load_pretrained_clf(pkl_model_path)

    def load_pretrained_clf(self, pkl_path) -> None:
        with open(pkl_path, "rb") as file:
            self.classifier = pickle.load(file)

    def get_embedings(self, url: str) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(url, padding=True, return_tensors="pt").to(
                self.device
            )
            return self.model(**tokens).pooler_output.cpu().detach().numpy()

    def get_batch_embeddings(self, urls: List[str], batch_size: int = 32) -> np.ndarray:
        # batching can be done outside
        batched_embeddings = []
        with torch.no_grad():
            for batch in batched(urls, batch_size):
                encoded_batch = self.tokenizer(
                    batch, padding=True, return_tensors="pt"
                ).to(self.device)
                model_output = self.model(**encoded_batch)
                batched_embeddings.append(model_output.pooler_output)

        return torch.cat(batched_embeddings, dim=0).detach().numpy()

    def prepare_result(self, probs: np.ndarray[float]) -> Dict[str, Union[str, float]]:
        # this won't work if we change class labels
        label = "dga" if probs[0] < probs[1] else "legit"
        return {
            "p_legit": probs[0],
            "p_dga": probs[1],
            "classification": label,
        }

    def predict(self, domain: str) -> Dict[str, Union[str, float]]:
        embeddings = self.get_embedings("http://" + domain)
        probs = self.classifier.predict_proba(embeddings)[0]
        return {domain: self.prepare_result(probs)}

    def predict_many(self, domain_list: List[str]) -> List[Dict]:
        http_urls_gen = ["http://" + url for url in domain_list]
        embeddings = self.get_batch_embeddings(http_urls_gen)
        probs = self.classifier.predict_proba(embeddings)
        results = {url: self.prepare_result(p) for url, p in zip(domain_list, probs)}
        return results
