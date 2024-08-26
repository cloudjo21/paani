from pydantic import BaseModel
from typing import List

from tunip import GSheet
from tweak.predict.predictor import PredictorConfig


class KeyPhrasePreProcConfig(BaseModel):
    pre_rank_overlap_threshold: float = 0.7


class KeyPhrasePostProcConfig(BaseModel):
    drop_phrases: List[str]
    drop_tokens: List[str]
    reject_candidate_keyphrase_pos_patterns: List[str]
    reject_candidate_keyphrase_str_patterns: List[str]
    allow_pos_patterns: List[str]
    allow_token_pos_pair_patterns: List[str]
    min_len_keyphrase_surface: int = 2
    max_len_keyphrase_surface: int = 20
    post_rank_overlap_threshold: float = 0.5


class KeyPhraseExtractorConfig(BaseModel):
    methods: List[str]
    """
    mss
    mmr_div_02
    mmr_div_05
    mmr_div_07
    """
    post_proc_config: KeyPhrasePostProcConfig
    predictor_config: PredictorConfig

    pre_proc_config: KeyPhrasePreProcConfig = KeyPhrasePreProcConfig(pre_rank_overlap_threshold=0.7)

    max_num_of_ngram: int = 10
    num_of_candidates: int = 50

    sentence_overlap_size: int = 10
    sentence_window_size: int = 40

    keyphrase_batch_size: int = 8

    class Config:
        arbitrary_types_allowed = True

class LoadDropTokenException(Exception):
    pass

class LoadDropPhraseException(Exception):
    pass

class LoadRejectStrPatternException(Exception):
    pass


def get_post_proc_config(gcs_keypath: str, gsheet_url: str, wsheet_names: dict):
    gsheet_client = GSheet(service_file=gcs_keypath)
    paani_setting_gsheet = gsheet_client.get_sheet(gsheet_url)
    try:
        drop_tokens = paani_setting_gsheet.worksheet_by_title(wsheet_names['drop_token']).get_as_df().iloc[:, 0].values.tolist()
    except:
        drop_tokens = None
        raise LoadDropTokenException()

    try:
        drop_phrases = paani_setting_gsheet.worksheet_by_title(wsheet_names['drop_phrase']).get_as_df().iloc[:, 0].values.tolist()
    except:
        drop_phrases = None
        raise LoadDropPhraseException()

    try:
        reject_str_patterns = paani_setting_gsheet.worksheet_by_title(wsheet_names['reject_str_pattern']).get_as_df().iloc[:, 0].values.tolist()
    except:
        reject_str_patterns = None
        raise LoadRejectStrPatternException()

    postproc_config = KeyPhrasePostProcConfig(

        # 절대 나올 필요가 없는 키워드
        drop_tokens=drop_tokens,
        drop_phrases=drop_phrases,
        reject_candidate_keyphrase_pos_patterns=[],
        reject_candidate_keyphrase_str_patterns=reject_str_patterns,
        allow_pos_patterns=['N', 'SN', 'SL', 'J', 'V', 'E'],
        allow_token_pos_pair_patterns=[]
    )
    return postproc_config
