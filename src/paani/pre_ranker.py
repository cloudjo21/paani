import operator

from typing import List

from paani import LOGGER
from paani import KeyPhrase, KeyPhraseList


class PreRanker:

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, key_phrase_list: KeyPhraseList) -> List[KeyPhrase]:
        key2score = dict()
        key2keyphrase = dict()
        for keyword in iter(key_phrase_list):
            score = key2score.get(keyword.surface, .0)
            key2score[keyword.surface] = keyword.score + score
            key2keyphrase[keyword.surface] = keyword
        keywords = []

        # sort by ascending order of score
        for k, v in sorted(key2score.items(), key=lambda item: item[1], reverse=False):
            key2keyphrase[k].score = v
            keywords.append(key2keyphrase[k])
        
        ranked = []
        rank = 1
        for i, keyphrase in enumerate(keywords):
            surface = keyphrase.surface
            overlapped = False
            for k in range(i+1, len(keywords)):
                next_surface = keywords[k].surface

                try:
                    if self.jaccard_sim(surface, next_surface) >= self.threshold:
                        overlapped = True
                        break
                except ZeroDivisionError as zde:
                    LOGGER.error(f"ZeroDivisionError is occurring for jaccard_sim between keyword: '{keyphrase}' and keyword: '{keywords[k]}'")
                    continue
            if not overlapped:
                keyphrase.rank = rank
                rank += 1
                ranked.append(keyphrase)

        # rerank by descending order of score
        ranked.sort(key=operator.attrgetter('score'), reverse=True)
        return ranked
    
    def jaccard_sim(self, a: str, b: str):
        s1 = set(list(a.replace(' ', '')))
        s2 = set(list(b.replace(' ', '')))
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))
