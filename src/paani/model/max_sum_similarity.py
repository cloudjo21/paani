import itertools
import operator
import torch

from typing import List

from paani import KeyPhrase, KeyPhraseList
from paani.calc_similarity import KeyPhraseSimilarityResult
from paani.model import KeyPhraseRanker


class MaxSumSimilarity(KeyPhraseRanker):

    def __init__(self, device, top_k=10, n_candidate=15):
        self.device = device
        self.top_k = top_k
        self.n_candidate = n_candidate

    def rank(self, key_phrase_list: KeyPhraseList, similarity_result: KeyPhraseSimilarityResult) -> List[KeyPhrase]:
        if len(key_phrase_list) < 1 or not similarity_result:
            return []

        candidate_matrix_index = similarity_result.doc2key_ranking[:self.n_candidate]
        candidate_phrases = [key_phrase_list[i] for i in candidate_matrix_index]
        candidate_phrase_list = KeyPhraseList(phrases=candidate_phrases)

        # key2key_similarity matrix must be diagonal matrix!
        ranked_key2key_similarity = similarity_result.key2key_similarity[candidate_matrix_index,:][:, candidate_matrix_index]

        # find minimum candidate combination including top-k keywords
        real_n_candidate = len(candidate_matrix_index)
        real_top_k = min(real_n_candidate, self.top_k)
        # TODO handle combinations to tensor
        key_combinations = torch.tensor(list(itertools.combinations(range(real_n_candidate), real_top_k)), dtype=torch.int64)
        key_combi_tensor = torch.zeros([key_combinations.shape[0], real_n_candidate] ).scatter_(1, key_combinations, 1.)
        key_combinations = key_combinations.to(self.device)
        key_combi_tensor = key_combi_tensor.to(self.device)
        key_combi_dot_similarity = torch.matmul(key_combi_tensor, ranked_key2key_similarity).sum(1)
        min_combi_index = key_combi_dot_similarity.argmin()
        chosen_keyphrase_indices = key_combinations[min_combi_index].tolist()

        ranked_key_phrases = [candidate_phrase_list[i] for i in chosen_keyphrase_indices]
        ranked_key_phrases.sort(key=operator.attrgetter('score'), reverse=True)
        return ranked_key_phrases
