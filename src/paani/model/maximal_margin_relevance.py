import operator

from typing import List

from paani import KeyPhrase, KeyPhraseList
from paani.calc_similarity import KeyPhraseSimilarityResult
from paani.model import KeyPhraseRanker


class MaximalMarginRelevance(KeyPhraseRanker):

    def __init__(self, diversity=0.5, top_k=10, n_candidate=50):
        self.top_k = top_k
        self.diversity = diversity
        self.n_candidate = n_candidate

    def rank(self, key_phrase_list: KeyPhraseList, similarity_result: KeyPhraseSimilarityResult) -> List[KeyPhrase]:
        if len(key_phrase_list) < 1 or not similarity_result:
            return []

        candidate_doc2key_index = similarity_result.doc2key_ranking[:self.n_candidate]
        ranked_doc2key_similarity = similarity_result.doc2key_similarity[candidate_doc2key_index]

        # candidate_index | shrinking indices | deflate in-context(in contrast to diversity)
        # maximal_index | increasing indices | inflate diversity
        # mmr argmax index
        maximal_key_index = [ranked_doc2key_similarity.argmax(dim=-1).item()]
        candidate_index = [i for i in range(len(candidate_doc2key_index)) if i != maximal_key_index[0]]

        real_n_candidate = len(candidate_index)
        real_top_k = min(real_n_candidate, self.top_k)

        for _ in range(real_top_k-1):
            doc2key_similarity = ranked_doc2key_similarity[candidate_index, None]
            key2key_similarity = similarity_result.key2key_similarity[candidate_index][:, maximal_key_index].max(dim=1).values

            mmr = (1 - self.diversity) * doc2key_similarity - self.diversity * key2key_similarity[:, None]
            mmr_max_index = candidate_index[mmr.argmax().item()]

            candidate_index.remove(mmr_max_index)
            maximal_key_index.append(mmr_max_index)
        
        ranked_key_phrases = [key_phrase_list[i] for i in maximal_key_index]
        ranked_key_phrases.sort(key=operator.attrgetter('score'), reverse=True)
        return ranked_key_phrases
