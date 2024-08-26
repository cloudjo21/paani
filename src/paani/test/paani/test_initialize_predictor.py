import unittest

from pathlib import Path

from tunip.env import NAUTS_HOME
from tunip.service_config import get_service_config

from tweak.predict.predictor import PredictorConfig
from tweak.predict.predictors import PredictorFactory

from paani import LOGGER
from paani.config import (
    KeyPhraseExtractorConfig,
    KeyPhrasePostProcConfig,
    LoadDropPhraseException,
    LoadRejectStrPatternException,
    get_post_proc_config
)


class IntializePredictorTest(unittest.TestCase):

    def setUp(self):
        self.service_config = get_service_config()

    def test_repetitive_loading_tokenizers(self):
        predictor_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "triton",
            "predict_output_type": "last_hidden.mean_pooling",
            "model_config": {
                "model_path": "/user/data/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "model_name": "monologg/koelectra-small-v3-discriminator",
                "remote_host": "0.0.0.0",
                "remote_port": "31016",
                "remote_model_name": "plm"
            },
            "tokenizer_config": {
                "model_path": "/user/data/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": "/user/data/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/vocab",
                "max_length": 128
            }
        }

        predictor_config = PredictorConfig.parse_obj(predictor_config_json)
        try:
            post_proc_config: KeyPhrasePostProcConfig = get_post_proc_config(
                gcs_keypath=str(Path(NAUTS_HOME) / 'resources' / f"{self.service_config.config.get('gcs.project_id')}.json"),
                gsheet_url='https://docs.google.com/spreadsheets/d/1r6gwmSyVJ8omxnuJaA5o_P8jYyjL-76vpjqjeI-P_WY/edit#gid=0',
                wsheet_names={
                    'drop_token': "불용키워드",
                    'reject_str_pattern': "차단키워드패턴",
                    'drop_phrase': "불용어구",
                }
            )
        except LoadDropPhraseException as ldpe:
            LOGGER.warning(str(ldpe))
            self.assertTrue()
            return
        except LoadRejectStrPatternException as lrspe:
            LOGGER.warning(str(lrspe))
            self.assertTrue()
            return

        extract_config = KeyPhraseExtractorConfig(
            # methods=['mss', 'mmr02','mmr05', 'mmr07'],
            methods=['mss', 'mmr05'],
            post_proc_config=post_proc_config,
            predictor_config=predictor_config
        )
        for _ in range(0, 20):
            predictor = PredictorFactory.create(extract_config.predictor_config)
            # just access to assert to initialize predictor
            predictor.tokenizer.tokenizer.vocab
