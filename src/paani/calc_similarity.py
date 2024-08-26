import torch

from pydantic import BaseModel
from typing import Optional


class KeyPhraseSimilarityResult(BaseModel):
    doc2key_similarity: torch.Tensor
    doc2key_ranking: torch.Tensor
    key2key_similarity: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True


class KeyPhraseSimilarityCalc:
    def __init__(self, calc_sim_among_keywords: bool = True):
        self.calc_sim_among_keywords = calc_sim_among_keywords

    def __call__(self, doc_embedding: torch.Tensor, key_embedding: torch.Tensor):
        cossim_doc = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cossim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        doc2key_similarity = cossim_doc(doc_embedding.reshape(1, -1), key_embedding)
        doc2key_ranking = doc2key_similarity.argsort(dim=-1, descending=True)

        result = None
        if self.calc_sim_among_keywords:
            key2key_similarity = cossim(key_embedding[:,None,:], key_embedding[None,:,:])
            # key2key_similarity = cossim(key_embedding, key_embedding)
            result = KeyPhraseSimilarityResult(doc2key_similarity=doc2key_similarity, doc2key_ranking=doc2key_ranking, key2key_similarity=key2key_similarity)
        else:
            result = KeyPhraseSimilarityResult(doc2key_similarity=doc2key_similarity, doc2key_ranking=doc2key_ranking)
        
        return result
