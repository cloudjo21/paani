from typing import List

from paani import LOGGER
from paani import KeyPhrase, KeyPhraseList


class PostRanker:
    """
    drop shorter overlapped keyword
    post_rank_strategy:
        - drop_shorter_overlapped
    """

    def __init__(self, threshold: float, drop_phrases = None):
        self.threshold = threshold
        self.drop_phrases = drop_phrases

    def __call__(self, key_phrase_list: KeyPhraseList) -> List[KeyPhrase]:
        key2score = dict()
        key2keyphrase = dict()
        for keyword in iter(key_phrase_list):
            score = key2score.get(keyword.surface, .0)
            key2score[keyword.surface] = keyword.score + score
            key2keyphrase[keyword.surface] = keyword
        keywords = []
        for k, v in sorted(key2score.items(), key=lambda item: item[1], reverse=True):
            key2keyphrase[k].score = v
            keywords.append(key2keyphrase[k])
        
        ranked = []
        rank = 1
        for i, keyphrase in enumerate(keywords):
            surface = keyphrase.surface

            overlapped = False
            for k in range(0, len(keywords)):
                if k == i:
                    continue
                next_surface = keywords[k].surface

                # if surface != next_surface and surface in next_surface:
                #     overlapped = True
                #     break
                try:
                    if self.jaccard_sim(surface, next_surface) >= self.threshold and len(surface) < len(next_surface):
                        overlapped = True
                        break
                except ZeroDivisionError as zde:
                    LOGGER.error(f"ZeroDivisionError is occurring for jaccard_sim between keyword: '{keyphrase}' and keyword: '{keywords[k]}'")
                    continue
            if not overlapped:

                # remove palindromic keyphrase like '영어교육 방법 한계 벗어나 영어'
                parts = surface.split(' ')
                if parts[0] in ' '.join(parts[1:]):
                    continue
                if parts[-1] in ' '.join(parts[0:-1]):
                    continue

                # ignore stopwords of phrase
                # TODO use abstract module to ignore them,
                #       and utilize specific handling stopwords like trie dic. or search engine
                if surface in self.drop_phrases:
                    continue

                keyphrase.rank = rank
                rank += 1
                ranked.append(keyphrase)

        return ranked

    def jaccard_sim(self, a: str, b: str):
        s1 = set(list(a.replace(' ', '')))
        s2 = set(list(b.replace(' ', '')))
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))
