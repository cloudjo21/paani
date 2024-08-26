import operator
import nltk
import re
import torch

from copy import deepcopy
from itertools import islice
from typing import List

from tunip.corpus_utils_v2 import AdjTagCorpusTokenMerger, CorpusRecord, CorpusToken
from tunip.nugget_api import Nugget
from tunip.re_utils import RejectPattern

from tweak.predict.predictor import Predictor
from tweak.predict.predictors import PredictorFactory

from paani import KeyPhrase, KeyPhraseList, LOGGER
from paani import RankResultForRanker, KeyPhraseRankResultDetail, FinalRankResult
from paani.calc_similarity import KeyPhraseSimilarityCalc, KeyPhraseSimilarityResult
from paani.config import KeyPhraseExtractorConfig
from paani.model.max_sum_similarity import MaxSumSimilarity
from paani.model.maximal_margin_relevance import MaximalMarginRelevance
from paani.post_ranker import PostRanker
from paani.pre_ranker import PreRanker
from paani.refine import KeywordRefinery


class KeyPhraseExtractor:
    pass


class KeyPhraseExtractorFactory:
    @classmethod
    def create(cls, ranker_name, device):
        if ranker_name == "mss":
            return MaxSumSimilarity(device, top_k=10, n_candidate=15)
        elif ranker_name == "mmr02":
            return MaximalMarginRelevance(diversity=0.2, top_k=10)
        elif ranker_name == "mmr05":
            return MaximalMarginRelevance(diversity=0.5, top_k=10)
        elif ranker_name == "mmr07":
            return MaximalMarginRelevance(diversity=0.7, top_k=10)
        else:
            raise Exception("Not Supported Ranker Method!")


class KeyBert(KeyPhraseExtractor):
    def __init__(self, config: KeyPhraseExtractorConfig):
        self.nugget = Nugget(split_sentence=True)

        # TODO refactor KeyPhrasePostProc
        # would be default values = ['N', 'SN', 'SL']
        self.drop_surfaces = config.post_proc_config.drop_tokens
        self.allow_pos_tags = config.post_proc_config.allow_pos_patterns
        self.allow_surface_pos_pairs = (
            config.post_proc_config.allow_token_pos_pair_patterns
        )
        self.reject_pos_pattern = RejectPattern(
            config.post_proc_config.reject_candidate_keyphrase_pos_patterns
        )
        self.reject_str_pattern = RejectPattern(
            config.post_proc_config.reject_candidate_keyphrase_str_patterns
        )

        # config for post-proc-filter
        self.min_len_keyphrase_surface = (
            config.post_proc_config.min_len_keyphrase_surface
        )
        self.max_len_keyphrase_surface = (
            config.post_proc_config.max_len_keyphrase_surface
        )

        self.max_len_ngram = config.max_num_of_ngram
        self.num_candidates = config.num_of_candidates

        # config for document vector
        self.sentence_overlap_size = config.sentence_overlap_size
        self.sentence_window_size = config.sentence_window_size

        self.keyphrase_batch_size = config.keyphrase_batch_size

        self.post_rank_overlap_threshold = (
            config.post_proc_config.post_rank_overlap_threshold
        )

        self.token_merger = AdjTagCorpusTokenMerger(
            allow_head_pos=["SN"],
            allow_pos_for_token_merge=["SN", "S"],
            merged_pos="SN",
        )

        doc_predictor_config = deepcopy(config.predictor_config)

        # :PreTrainedModelPredictor
        self.predictor: Predictor = PredictorFactory.create(config.predictor_config)

        # TODO set individual document predictor config into dag.yml and then read it
        doc_predictor_config.predict_output_type = "last_hidden.global_mean_pooling"
        self.doc_predictor: Predictor = PredictorFactory.create(doc_predictor_config)

        self.key_phrase_rankers = dict()
        for method in config.methods:
            ranker = KeyPhraseExtractorFactory.create(
                method, config.predictor_config.device
            )
            self.key_phrase_rankers[method] = ranker

        self.post_ranker = PostRanker(
            threshold=self.post_rank_overlap_threshold,
            drop_phrases=config.post_proc_config.drop_phrases
        )
        self.pre_ranker = PreRanker(threshold=0.6)

    def __call__(self, text: str, given_keywords: KeyPhraseList = None):
        text = re.sub(r"\s+", " ", text.strip())
        sents = text.split("\n")

        nuggets: List[CorpusRecord] = self.nugget.record_v2(sents)
        if not nuggets:
            # TODO define response class that has final_ranked and rank_result_detail using Optional
            return None, None

        # token merge for number and symbol
        nuggets = [self.token_merger(n) for n in nuggets]

        # generate candidates
        candidate_phrases: KeyPhraseList = (
            self._ngram_phrase_result(nuggets) if not given_keywords else given_keywords
        )

        candidate_phrases = self.post_ranker(candidate_phrases)

        similarity_result: KeyPhraseSimilarityResult = self._calc_similarity(
            candidate_phrases, text
        )

        ranked_phrase_list = self.rank_by_doc2key(candidate_phrases, similarity_result)

        rankers_result: List[KeyPhraseList] = []
        rank_results = []
        for name, ranker in self.key_phrase_rankers.items():
            result: KeyPhraseList = ranker.rank(ranked_phrase_list, similarity_result)
            # all the results
            rankers_result.append(result)
            # result for-each-ranker
            rank_results.append(RankResultForRanker(ranker_name=name, ranking=result))
        rank_result_detail = KeyPhraseRankResultDetail(results=rank_results)

        # post-proc for filtering pattern
        post_ranked = list()
        for ranker_result in rankers_result:
            for keyphrase_list in ranker_result:
                post_ranked.append(keyphrase_list)

        post_keyphrase_list = KeyPhraseList(phrases=post_ranked)

        # post-proc for overlapped key phrases from various rankers
        final_ranked = FinalRankResult(ranking=self.post_ranker(post_keyphrase_list))

        return final_ranked, rank_result_detail

    def call_v2(
        self, text: str, refinery: KeywordRefinery, given_keywords: KeyPhraseList = None
    ):
        nuggets = self.preprocess(text)
        candidate_phrases = self.make_candidates(nuggets, given_keywords)

        clean_phrases = self.drop_duplicates(refinery, candidate_phrases) if refinery else candidate_phrases
        similarity_result, ranked_phrase_list = self.calc_similarity_and_rank(
            clean_phrases, text
        )
        rank_result_detail: KeyPhraseRankResultDetail = self.rank(
            similarity_result, ranked_phrase_list
        )
        rank_result_final = self.post_rank(rank_result_detail)

        return rank_result_final, rank_result_detail

    def preprocess(self, text: str) -> List[CorpusRecord]:
        text = re.sub(r"[ ]+", " ", text.strip())
        sents = text.split("\n")

        nuggets: List[CorpusRecord] = self.nugget.record_v2(sents)
        if not nuggets:
            # TODO define response class that has final_ranked and rank_result_detail using Optional
            return None, None

        # token merge for number and symbol
        nuggets = [self.token_merger(n) for n in nuggets]
        return nuggets

    def make_candidates(
        self, nuggets: List[CorpusRecord], given_keywords: KeyPhraseList = None
    ) -> KeyPhraseList:
        candidate_phrases: KeyPhraseList = (
            self._ngram_phrase_result(nuggets) if not given_keywords else given_keywords
        )
        return candidate_phrases

    def drop_duplicates(
        self, refinery: KeywordRefinery, candidate_phrases: KeyPhraseList
    ) -> KeyPhraseList:
        # refine candidate phrases
        refined_results_for_candi = [
            refinery.get_or_not(c.surface) for c in candidate_phrases
        ]
        phrases_scored = []
        for c, r in zip(candidate_phrases, refined_results_for_candi):
            if r.success is False or (r.success is True and (r.first_term_pass ^ r.last_term_pass)):
                continue
            c.score = r.mle
            phrases_scored.append(c)
        phrases_scored.sort(key=operator.attrgetter("score"), reverse=False)
        refined_phrases = KeyPhraseList(phrases=phrases_scored)
        return self.pre_ranker(refined_phrases)

    def calc_similarity_and_rank(self, candidate_phrases, text) -> tuple:
        similarity_result: KeyPhraseSimilarityResult = self._calc_similarity(
            candidate_phrases, text
        )
        ranked_phrase_list = self.rank_by_doc2key(candidate_phrases, similarity_result)
        return similarity_result, ranked_phrase_list

    def rank(
        self,
        similarity_result: KeyPhraseSimilarityResult,
        ranked_phrase_list: KeyPhraseList,
    ) -> KeyPhraseRankResultDetail:
        rankers_result: List[KeyPhraseList] = []
        rank_results = []
        for name, ranker in self.key_phrase_rankers.items():
            result: KeyPhraseList = ranker.rank(ranked_phrase_list, similarity_result)
            # all the results
            rankers_result.append(result)
            # result for-each-ranker
            rank_results.append(RankResultForRanker(ranker_name=name, ranking=result))
        rank_result_detail = KeyPhraseRankResultDetail(results=rank_results)
        return rank_result_detail

    def post_rank(
        self, rank_result_detail: KeyPhraseRankResultDetail
    ) -> FinalRankResult:
        # post-proc for filtering pattern
        post_ranked = list()
        for ranker_result in rank_result_detail.results:
            post_ranked.extend(ranker_result.ranking)

        post_keyphrase_list = KeyPhraseList(phrases=post_ranked)

        # post-proc for overlapped key phrases from various rankers
        final_ranked = FinalRankResult(ranking=self.post_ranker(post_keyphrase_list))

        return final_ranked

    def rank_by_doc2key(
        self, key_phrases: KeyPhraseList, similarity_result: KeyPhraseSimilarityResult
    ) -> KeyPhraseList:
        """
        rank key phrases corresponding to text
        """
        if not similarity_result:
            return KeyPhraseList(phrases=[])

        ranked_phrase_list = []
        # for rank, index_tensor in enumerate(similarity_result.doc2key_ranking[:self.num_candidates]):
        for rank, index_tensor in enumerate(similarity_result.doc2key_ranking):
            index = index_tensor.item()
            phrase: KeyPhrase = key_phrases[index]
            phrase.rank = rank + 1
            phrase.score = similarity_result.doc2key_similarity[index].item()
            ranked_phrase_list.append(phrase)
        return KeyPhraseList(phrases=ranked_phrase_list)

    def _ngram_phrase_result(self, nuggets: List[CorpusRecord]):
        # generate candidates
        phrases = []
        for nugget in nuggets:
            candi_tokens = []
            for nugget_token in nugget.tokens:
                if self._allow_nugget_token(nugget_token):
                    candi_tokens.append(nugget_token)
                else:
                    self._extend_ngram_phrase(phrases, candi_tokens)
                    candi_tokens = []

            self._extend_ngram_phrase(phrases, candi_tokens)
            # TODO cut-off phrases by maximum number of ngram candidate

        return KeyPhraseList(phrases=phrases)

    def _extend_ngram_phrase(self, phrases, candi_tokens):
        if candi_tokens:

            if candi_tokens[0].pos in ["J", "V", "E"] or candi_tokens[-1].pos != "N":
                return

            # make ngram
            token_ngrams = nltk.everygrams(
                candi_tokens, max_len=min(len(candi_tokens), self.max_len_ngram)
            )

            candidate_phrases = []
            # filter candiates
            for tokens in token_ngrams:
                if tokens[0].pos in ["J", "V", "E"]:
                    continue
                if tokens[-1].pos in ["J"]:
                    continue
                phrase = KeyPhrase(tokens=tokens)
                if self._allow_keyphrase(phrase):
                    candidate_phrases.append(phrase)

            phrases.extend(candidate_phrases)

    def _allow_nugget_token(self, nugget_token: CorpusToken):
        if nugget_token.surface in self.drop_surfaces:
            return False
        if nugget_token.pos in self.allow_pos_tags:
            return True
        elif (
            f"{nugget_token.surface}/{nugget_token.pos}" in self.allow_surface_pos_pairs
        ):
            return True
        else:
            return False

    def _calc_similarity(self, key_phrases: KeyPhraseList, text: str):
        if len(key_phrases) < 1:
            return None

        keyphrase_embeddings = self._infer_keyphrase_embeddings(key_phrases)
        document_embedding = self._infer_doc_embedding(text)

        sim_calc = KeyPhraseSimilarityCalc()
        sim_calc_result: KeyPhraseSimilarityResult = sim_calc(
            document_embedding, keyphrase_embeddings
        )
        return sim_calc_result

    def _infer_doc_embedding(self, text: str):
        tokens = text.split()
        sentences = [
            tokens[
                i
                - self.sentence_overlap_size : i
                + self.sentence_window_size
                + self.sentence_overlap_size
            ]
            for i in range(
                self.sentence_overlap_size, len(tokens), self.sentence_window_size
            )
        ]
        if not sentences:
            sentences = [text]

        embeddings = []
        for batch in self._batched(sentences, self.keyphrase_batch_size):
            embeddings.append(self.doc_predictor.predict(list(batch)))
        return torch.mean(torch.vstack(embeddings), axis=0)

    @staticmethod
    def _batched(iterable, n):
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    def _infer_keyphrase_embeddings(self, key_phrases: KeyPhraseList):
        batch_embeddings = []
        for phrases in self._batched(key_phrases, self.keyphrase_batch_size):
            batch_embeddings.append(
                self.predictor.predict([p.surface for p in phrases])
            )
        keyphrase_embeddings = torch.vstack(batch_embeddings)
        return keyphrase_embeddings

    def _allow_keyphrase(self, keyphrase: KeyPhrase):
        surface = keyphrase.surface
        if self.reject_pos_pattern.match(surface):
            return False
        if self.reject_str_pattern.match(surface):
            return False
        if (
            self.min_len_keyphrase_surface
            < len(surface)
            < self.max_len_keyphrase_surface
        ):
            return True
        return False
