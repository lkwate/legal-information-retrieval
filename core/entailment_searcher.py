from .document_factory import DocumentFactor
from pyserini.dsearch import SimpleDenseSearcher, AutoQueryEncoder
from pyserini.search import SimpleSearcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import os
from loguru import logger


class EntailmentSearcher:
    def __init__(
        self,
        model,
        dense_index,
        sparse_index,
        documents,
        hidden_dim=768,
        top_k=5,
        device="cpu",
    ) -> None:
        self.document_factory = DocumentFactor(documents)
        self.encoder = AutoQueryEncoder(
            encoder_dir=model, pooling="mean", l2_norm=True, device=device
        )
        self.dense_searcher = SimpleDenseSearcher(
            query_encoder=self.encoder, index_dir=dense_index
        )
        self.sparse_searcher = SimpleSearcher(sparse_index)
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.cross_search_prefix = "cross_"
        self.pairwise_search_prefix = "pairwise_"

    def __call__(self, query: str):
        hits = self.sparse_searcher.search(query, k=self.top_k + 1)[1:]
        output = []

        for hit in hits:
            logger.info(f"entailment search for the hit : {hit.docid}")
            try:
                case = self.document_factory.get_document(str(hit.docid))
                # cross_out = self._cross_search(query, case)
                pairwise_out = self._pairwise_search(query, case)
                # result = {"score": hit.score, **cross_out, **pairwise_out}
                result = {"score": hit.score, **pairwise_out}
                output.append(result)
            except ValueError:
                continue
        return output

    def _join_paragraph(self, index: int, paragraphs: List[str]):
        if index >= len(paragraphs):
            raise ValueError("index of out of paragraph bounds")
        content = os.linesep.join(
            [par for i, par in enumerate(paragraphs) if i != index]
        )
        return content

    def _cross_search(self, query: str, case: str):
        logger.info("cross search started...")
        query_encoded = self.encoder.encode(query)
        case_encoded = self.encoder.encode(case)
        base = cosine_similarity(
            query_encoded.reshape(1, self.hidden_dim),
            case_encoded.reshape(1, self.hidden_dim),
        )

        query_paragraphs = [par for par in query.split("\n") if par]
        case_paragraphs = [par for par in case.split("\n") if par]

        query_encoded_paragraphs = [
            self.encoder.encode(self._join_paragraph(i, query_paragraphs))
            for i in range(len(query_paragraphs))
        ]
        case_encoded_paragraphs = [
            self.encoder.encode(self._join_paragraph(i, case_paragraphs))
            for i in range(len(case_paragraphs))
        ]

        query_scores = list(
            map(
                lambda q1, hidden_dim=self.hidden_dim, query_encoded=query_encoded: cosine_similarity(
                    q1.reshape(1, hidden_dim), query_encoded.reshape(1, hidden_dim)
                ),
                query_encoded_paragraphs,
            )
        )
        case_scores = list(
            map(
                lambda q2, hidden_dim=self.hidden_dim, case_encoded=case_encoded: cosine_similarity(
                    q2.reshape(1, hidden_dim), case_encoded.reshape(1, hidden_dim)
                ),
                case_encoded_paragraphs,
            )
        )
        query_par_index = np.argmin(query_scores)
        case_par_index = np.argmin(case_scores)
        best_pair = (query_par_index, case_par_index)
        output = {
            f"{self.cross_search_prefix}pair": best_pair,
            f"{self.cross_search_prefix}query_paragraph": query_paragraphs[query_par_index],
            f"{self.cross_search_prefix}case_paragraph": case_paragraphs[case_par_index],
            f"{self.cross_search_prefix}query_scores": query_scores,
            f"{self.cross_search_prefix}case_scores": case_scores,
        }
        logger.info("cross search completed.")
        return output

    def _pairwise_search(self, query: str, case: str):
        logger.info("pairwise search started...")
        query_paragraphs = [par for par in query.split("\n") if par]
        case_paragraphs = [par for par in case.split("\n") if par]

        query_encoded_paragraphs = list(map(self.encoder.encode, query_paragraphs))
        case_encoded_paragraphs = list(map(self.encoder.encode, case_paragraphs))

        best_score = -2
        best_pair = None
        for query_par_index, query_par in enumerate(query_encoded_paragraphs):
            for case_par_index, case_par in enumerate(case_encoded_paragraphs):
                sim_score = cosine_similarity(
                    query_par.reshape(1, self.hidden_dim),
                    case_par.reshape(1, self.hidden_dim),
                )
                if sim_score > best_score:
                    best_score = sim_score
                    best_pair = (query_par_index, case_par_index)
        output = {
            f"{self.pairwise_search_prefix}pair": best_pair,
            f"{self.pairwise_search_prefix}query_paragraph": query_paragraphs[best_pair[0]],
            f"{self.pairwise_search_prefix}case_paragraph": case_paragraphs[best_pair[1]],
        }
        logger.info("pairwise search completed.")
        return output
