import math
import nltk

from typing import Optional
from pydantic import BaseModel

from paani import LOGGER
from paani.refine import (
    COND_ENTROPY_MEAN_LEN,
    COND_ENTROPY_MLE_THRESHOLD,
    COND_ENTROPY_MLE_WORKAROUND_THRESHOLD,
    KeywordRefinery
)


class EitherBoundaryPassViolationException(Exception):
    pass


class ConditionalEntropyRefineryResponse(BaseModel):
    keyword: str
    first_term_pass: bool
    last_term_pass: bool
    success: bool = False
    mle: Optional[float]
    mle_workaround: Optional[float] = None

    def accept_or_not(self):
        return self.success and (self.mle > 0.0003)


class ConditionalEntropyRefinery(KeywordRefinery):

    def __init__(
        self,
        token2count,
        count2token_num,
        cooccur2count,
        count2cooccur_num,
        N_token_count_sum,
        boundary_terms_refine_threshold={},
        T_token=5,
        T_cooccur=6
    ):
        self.token2count = token2count
        self.count2token_num = count2token_num

        self.cooccur2count = cooccur2count
        self.count2cooccur_num = count2cooccur_num

        self.N_token_count_sum = N_token_count_sum

        self.boundary_terms_refine_threshold = boundary_terms_refine_threshold

        self.T_token = T_token
        self.T_cooccur = T_cooccur


    def _d_count_of_token(self, count):
        return ((count+1) * self.count2token_num[count + 1]) / self.count2token_num[count]

    def _d_count_of_cooccur(self, count):
        return ((count+1) * self.count2cooccur_num[count + 1]) / self.count2cooccur_num[count]


    def _count_of_bigram(self, bigrams):
        counts = []
        n_vocabs = []
        for token in bigrams:
            count = self.token2count.get(token) or 1
            if count < self.T_token:
                r_count = self._d_count_of_token(count)
            else:
                r_count = count
            counts.append(r_count)

            n_vocab = self.count2token_num.get(count) or (self.count2token_num.get(count-1) or 0.0)
            n_vocabs.append(n_vocab)

        return counts, n_vocabs

    def _count_of_bigram_cooccur(self, co_occurrences):

        counts = []
        n_vocabs = []
        for co_occur in co_occurrences:
            count = self.cooccur2count.get(co_occur) or 1
            if count < self.T_cooccur:
                r_count = self._d_count_of_cooccur(count)
            else:
                r_count = count
            counts.append(r_count)

            n_vocab = self.count2cooccur_num.get(count) or (self.count2cooccur_num.get(count-1) or 0.0)
            n_vocabs.append(n_vocab)

        return counts, n_vocabs
    

    def __call__(self, keyword, verbose=False):
        condensed = keyword.replace(' ', '')
        if len(condensed) < 6:
            chars = '^' + condensed
        else:
            chars = '^' + condensed + '$'
        bigrams = [b[0] + b[1] for b in list(nltk.bigrams(chars))]

        co_occurrences = ['_'.join([a,b]) for a, b in zip(bigrams[:-1], bigrams[1:])]
        cooccur_counts, _ = self._count_of_bigram_cooccur(co_occurrences=co_occurrences)
        token_counts, _ = self._count_of_bigram(bigrams=bigrams)

        cond_entropy_terms = [
            (co_cnt / self.N_token_count_sum) * math.log((co_cnt / self.N_token_count_sum)/(tok_cnt / self.N_token_count_sum)) for co_cnt, tok_cnt in zip(cooccur_counts, token_counts[:-1])
        ]
        cond_entropy = -1 * math.fsum([p for p in cond_entropy_terms])

        LOGGER.debug(f"bigrams: {bigrams}")
        LOGGER.debug(f"entropy_terms: {cond_entropy_terms}")

        if verbose:
            return cond_entropy, cond_entropy_terms
        else:
            return cond_entropy

    def get_or_not(self, keyword) -> ConditionalEntropyRefineryResponse:
        cond_entropy, cond_entropy_terms = self.__call__(keyword, verbose=True)

        len_kwd = len(keyword.replace(' ', ''))

        try:
            first_thld_map = self.boundary_terms_refine_threshold.get('first') or None
            if first_thld_map:
                first_thld = first_thld_map.get(str(len_kwd)) or 0.0
                if first_thld > 0.0:
                    first_term_pass = cond_entropy_terms[0] < -1 * (first_thld_map.get(str(len_kwd)) or 0.0) 
                else:
                    first_term_pass = False
            else:
                first_term_pass = False

            last_thld_map = self.boundary_terms_refine_threshold.get('last') or None
            if last_thld_map:
                last_thld = last_thld_map.get(str(len_kwd)) or 0.0
                if last_thld > 0.0:
                    last_term_pass = cond_entropy_terms[-1] < -1 * (last_thld_map.get(str(len_kwd)) or 0.0)
                else:
                    last_term_pass = False
            else:
                last_term_pass = False
            
            neither_pass = not first_term_pass and not last_term_pass
            if neither_pass:
                response = ConditionalEntropyRefineryResponse(
                    keyword=keyword,
                    first_term_pass=first_term_pass,
                    last_term_pass=last_term_pass,
                    mle=cond_entropy
                )
                return response

            either_pass = first_term_pass ^ last_term_pass
            if either_pass:
                if not first_term_pass:
                    boundary_thsld = first_thld_map.get(str(len_kwd)) or 0.0
                    boundary_abs_cond_entropy = abs(cond_entropy_terms[0])
                elif not last_term_pass:
                    boundary_thsld = last_thld_map.get(str(len_kwd)) or 0.0
                    boundary_abs_cond_entropy = abs(cond_entropy_terms[-1])
                else:
                    raise EitherBoundaryPassViolationException()

                if boundary_thsld > 0.0:
                    mle_workaround = (boundary_abs_cond_entropy / boundary_thsld) * math.log(cond_entropy / COND_ENTROPY_MLE_THRESHOLD) * min(1, (1 + math.log(len_kwd / COND_ENTROPY_MEAN_LEN)))
                    if mle_workaround > COND_ENTROPY_MLE_WORKAROUND_THRESHOLD:
                        response = ConditionalEntropyRefineryResponse(
                            keyword=keyword,
                            first_term_pass=first_term_pass,
                            last_term_pass=last_term_pass,
                            success=True,
                            mle=cond_entropy,
                            mle_workaround=mle_workaround
                        )
                        return response 
                    else:
                        response = ConditionalEntropyRefineryResponse(
                            keyword=keyword,
                            first_term_pass=first_term_pass,
                            last_term_pass=last_term_pass,
                            success=False,
                            mle=cond_entropy,
                            mle_workaround=mle_workaround
                        )
                        return response
                else:
                    response = ConditionalEntropyRefineryResponse(
                        keyword=keyword,
                        first_term_pass=first_term_pass,
                        last_term_pass=last_term_pass,
                        success=False,
                        mle=cond_entropy
                    )
                    return response
            else:
                # if both the first term and last one are passed
                response = ConditionalEntropyRefineryResponse(
                    keyword=keyword,
                    first_term_pass=first_term_pass,
                    last_term_pass=last_term_pass,
                    success=True,
                    mle=cond_entropy
                )
                return response
        except IndexError:
            LOGGER.warning(f'KEYWORD:{keyword} has such conditional entropy terms: {cond_entropy_terms}')
            response = ConditionalEntropyRefineryResponse(
                keyword=keyword,
                first_term_pass=False,
                last_term_pass=False,
                success=False,
                mle=cond_entropy
            )
            return response
