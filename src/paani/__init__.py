import logging

from pydantic import BaseModel
from typing import List, Optional

from tunip.corpus_utils_v2 import CorpusToken, merge_surface_filter_suffix
from tunip.logger import init_logging_handler


LOGGER = init_logging_handler(name="paani", level=logging.INFO)


class KeyPhrase(BaseModel):
    tokens: List[CorpusToken]
    rank: int = -1
    score: float = 0.0
    token_separator: str = ' '

    def __str__(self) -> str:
        return f"keyword: {self.surface}, rank: {self.rank}, score: {self.score}"
    
    @property
    def surface(self):
        return merge_surface_filter_suffix(self.tokens, self.token_separator)


class KeyPhraseList(BaseModel):
    phrases: List[KeyPhrase]

    def __getitem__(self, i):
        return self.phrases[i]

    def __iter__(self):
        return iter(self.phrases)

    def __len__(self):
        return len(self.phrases)

    # def __repr__(self):
    #     return "\n".join([str(p) for p in self.phrases])


class RankResultForRanker(BaseModel):
    ranker_name: str
    ranking: List[KeyPhrase]

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return f'ranker_name: {self.ranker_name}\n' + '\t\n'.join([str(r) for r in self.ranking])

class KeyPhraseRankResultDetail(BaseModel):
    results: List[RankResultForRanker]

    class Config:
        arbitrary_types_allowed = True

    def pretty(self):
        return '\n'.join([str(r) for r in self.results])


class RankingEntity(BaseModel):
    keyword: str
    rank: int
    score: float


class FinalRankResult(BaseModel):
    ranking: List[KeyPhrase]

    class Config:
        arbitrary_types_allowed = True

    def pretty(self):
        return '\n'.join([str(r) for r in self.ranking])

    def entities(self) -> List[RankingEntity]:
        return [RankingEntity(keyword=k.surface, rank=k.rank, score=k.score) for k in self.ranking]
