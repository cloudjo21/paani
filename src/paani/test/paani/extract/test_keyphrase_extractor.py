import unittest
import urllib.parse

from tunip.path.mart import (
    MartPretrainedModelPath,
    MartTokenizerPath
)
from tunip.service_config import get_service_config
from tweak.predict.predictor import PredictorConfig

from paani.config import (
    KeyPhraseExtractorConfig,
    KeyPhrasePostProcConfig,
)
from paani.extract.keyphrase_extractor import KeyBert


class KeyPhraseExtractorTest(unittest.TestCase):

    def setUp(self):
        service_config = get_service_config()

        model_name='monologg/koelectra-small-v3-discriminator'
        # model_name='snunlp/KR-ELECTRA-discriminator'
        # model_name = 'klue/roberta-base'

        quoted_model_name = urllib.parse.quote(model_name, safe='')
        model_nauts_path = MartPretrainedModelPath(
            user_name=service_config.username,
            model_name=quoted_model_name
        )
        tokenizer_nauts_path = MartTokenizerPath(
            user_name=service_config.username,
            tokenizer_name=quoted_model_name
        )
        plm_model_path = str(model_nauts_path)
        tokenizer_path = f"{plm_model_path}/vocab"

        self.predictor_config_json = {
            "predict_tokenizer_type": "auto",
            "predict_model_type": "torchscript",
            "predict_output_type": "last_hidden.mean_pooling",
            "predict_model_config": {
                "model_path": f"/user/{service_config.username}/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/torchscript",
                "model_name": "monologg/koelectra-small-v3-discriminator",
                "encoder_only": False
            },
            "tokenizer_config": {
                "model_path": f"/user/{service_config.username}/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": f"/user/{service_config.username}/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/vocab",
                # "task_name": None,
                "max_length": 128
            }
        }
        # self.predictor_config_json = {
        #     "predict_tokenizer_type": "auto",
        #     "predict_model_type": "torchscript",
        #     # "predict_model_type": "triton",
        #     # "predict_model_type": "auto",
        #     "predict_output_type": "last_hidden.mean_pooling",
        #     # "predict_output_type": "last_hidden_with_attention_mask",
        #     # "predict_output_type": "last_hidden_with_attention_mask.mean_pooling",
        #     "zero_padding": False,
        #     "model_config": {
        #         "model_path": f"{plm_model_path}/torchscript",
        #         # "model_path": f"{plm_model_path}",
        #         "model_name": model_name,
        #         "remote_host" : "0.0.0.0",
        #         "remote_port" : 31016,
        #         "remote_model_name" : "plm",
        #     },
        #     "tokenizer_config": {
        #         "model_path": plm_model_path,
        #         "path": tokenizer_path,
        #         # XXX "path": tokenizer_path,
        #         "max_length": 128
        #     }
        # }

    def test_rank(self):
        # text = """아이들 눈높이에서 아이들을 보시는 친절한 선생님이세요.  동화구연 으로 책읽기 놀이 돌봄수업, 한글 수업을 추천해요. """
        text = """예중 예고 준비, 미대 준비를 통해 미술수업이 어떻게 운영되어야 하는지잘 알고 있습니다. 또 1년간 진행했던 장애인 미술 교육 봉사 경험으로  다양한 미술 재료 사용, 창의적인 주제 등을  수업에 재미있게 녹일 수 있습니다. 매 수업마다 어떤 태도로 임해야하는지, 어떻게 수업 분위기를 이끌 것인지에 대해 늘 책임감을 가지고 있고, 이러한 저의 경험과 생각을 바탕으로 수업을 진행하려 합니다. 나아가 대학 수업에서 배우는 다소 딱딱할 수 있는 이론들을(화가, 작업,서양미술사) 아이들에게 재미있게 소개할 예정입니다! 아이들이 예술의 재미를 더욱 느낄 수 있도록 노력하겠습니다! """
        # text = """영국에서 4년동안 거주한경험과 초등학생 동생에게 독해와 스피킹을 할 수 있도록 1년간 도움을 주었습니다! 아이들이 단순히 외우는 영어가 아닌 이해하기 편하게 접근하는 영어와 귀가 뚫리면서 시작부터 영어에 거부감이 생기지 않도록 도와주고싶습니다. 또한 유치원/학교에서의 영어숙제와 토플주니어도 준비하고있다면 같이 준비하고 싶습니다"""
        print(self.predictor_config_json)

        postproc_config = KeyPhrasePostProcConfig(
            drop_tokens=[],
            drop_phrases=[],
            reject_candidate_keyphrase_pos_patterns=[],
            reject_candidate_keyphrase_str_patterns=['동안$'],
            allow_pos_patterns=['N', 'SN', 'SL', 'J', 'V', 'E'],
            # allow_pos_patterns=['N', 'SN', 'SL'],
            allow_token_pos_pair_patterns=[]
        )
        predictor_config = PredictorConfig.parse_obj(self.predictor_config_json)
        config = KeyPhraseExtractorConfig(methods=['mss'], post_proc_config=postproc_config, predictor_config=predictor_config)

        ranked_keywords, _ = KeyBert(config)(text)
        print(str(ranked_keywords.pretty()))
        assert ranked_keywords is not None

    def test_rank_longer_text(self):
        texts = [
            """재미있는 과학실험들 함께 해보아요~!!!!!!!!!""",
            """최근까지 수학/과학을 하고 있는 학생으로서 아이들이 어떤 점을 어려워하는지, 이를 어떻게 극복해내야 하는지를 누구보다 잘 이해하고 있습니다.""",
            """저는 수학을 전공했고, 수학 과외, 진로 멘토링, 수학 학습 멘토링, 창의 수학 수업 등을 진행하고 있습니다. \n\n요즘 초등학생 때부터 수학을 포기하는 친구들이 많아요 ಥ_ಥ  초등학교 수학 교육은 아이들로 하여금 수학에 대한 흥미를 잃지 않도록 하는 것이 가장 중요합니다! 2019년 저와 함께했던 5학년 친구는 처음 만났을 때 수학 성적이 50점이고 수학을 제일 싫어한다고 말했습니다. 하지만 지금은 매번 90점 이상의 점수를 받아오고, 수학을 즐기며 공부합니다. \n그러려면 학생 수준에 맞는 교재를 선택하는 것이 중요합니다! 다양한 연령대, 수준의 학생들을 교육하면서 쌓은 경험을 토대로 학생의 수준에 맞는 교재를 선택하고 지도하여 수학을 좋아하는 아이로 성장하도록 돕겠습니다!""",
            """먼저 지역아동센터에서 아이들에게 수학을 멘토링한 경험이 있고, 초등학교에서 교육봉사로 보조 교사 역할을 수행한 경험이 있습니다. 아이들이 학습에 집중을 하기 어려워 할 때나 흥미를 느끼지 못할 때 등을 경험해본 적이 있고 이를 해결하기 위한 스스로의 교육법도 수많이 고민해보았습니다.\n우선 저의 학생이  된 아이와 앞으로의 공부계획을 ‘함께’ 세워볼 생각입니다. 공부할 내용을 아이와 같이 알아보고 계획하면서 아이의 성향을 파악하여 제가 도움을 주어야 하는 부분을 점검할 것입니다. 또 진도표와 숙제점검표 등을 사용하여 그날의 수행도를 기록하고, 멘티가 멘토와의 약속을 성실히 지킨 것에 대한 충분한 칭찬과 보상을 해줄 생각입니다. \n 저는 모든 수업이 끝난 뒤, 반드시 그날 배운 내용을 아이가 직접 저에게 설명해보도록 할 예정입니다. 이는 실제로 제가 중고등학교 시절 공부했던 방법으로 본인의 이해 정도를 파악하는데 많은 도움을 주었습니다. 학습한 내용을 남에게 설명해보는 과정을 통해 아이가 자신이  무엇을 알고, 무엇을 모르는지 스스로 인지하도록 할 계획입니다. 또 이 과정은 학습 내용을 본인만의 언어로 한 번 더 정리하는 과정이기 때문에 멘티의 학습능력을 향상하는데 도움을 주는 가장 중요한 시간이라고 생각합니다.\n 수업시간 전엔 EBS강의 등을 통해 학습할 내용을 정확히 파악할 예정입니다. 또 도움이 될만한 자료를 찾거나 흥미를 유발할 수 있는 소재들을 준비하는 등 수업 준비를 철저하게 진행할 것입니다.""",
            """초등학생 및 중학생 대상 수학 멘토로 활동. 청심국제고등학교 재학 당시 수학 및 통계 과목에서 매번 교과우수상 수상.\n수학에서 (특히 초등학교 수학) 가장 중요한 것은 1) 개념 정리 및 원리 이해, 2) 특정 개념이 중학교 수학과 어떻게 연관되어 이어지고 심화되는지 이해하기, 3) 오답에 대해서는 틀린 이유와 원리를 이해하고 반복하여 풂으로써 자연스레 응용된 문제도 풀 수 있는 ‘생각하는 능력’ 향상하기라고 생각합니다. 수학 유형을 암기하기만 한다면 중고등학교 응용 수학에서 어려움이 생길 수밖에 없습니다. 그래서 저는 크게 두 가지에 집중할 예정입니다. 첫 번째는 수학적 개념/원리를 (중학교 교과와도 어떻게 이어지고 응용되는지를 포함) 이해하기 쉽게 정리된 노트로 만들어 학생들과 공유할 예정입니다. 두 번째로는, 오답 복습인데요, 아이가 오답 노트를 만들도록 하여 틀린 이유와 틀린 문제 뒤에 숨겨진 원리를 짧게라도 스스로 적어보는 시간을 갖는 것입니다. 정말 틀린 이유를 완벽하게 소화했는지 알기 위해서는 ‘같은 유형, 다른 문제’를 풀어보는 것이라고 생각하기 때문에 그 다음 수업 때 비슷한 유형의 또다른 문제를 반복적으로 아이에게 제공할 계획입니다! 4) 매주 복습 테스트를 진행하여 아이가 약한 부분이 무엇인지 파악할 예정입니다:)"""
            # """제안이유
    
            # 대한민국의 관문공항인 인천국제공항과 연계되어 지난 2001년과 2009년 각각 개통된 인천국제공항고속도로와 인천대교는 당초 공항으로부터의 원활한 교통소통을 위한 전용의 고속도로로 계획이 입안되었음에도 불구하고, 건설비용에 대한 정부의 재정부담을 이유로 「사회기반시설에 대한 민간투자법」에 따른 민자유치사업으로 전환되어 동법에 따른 협약의 적용을 받는 사회간접자본시설로서 오늘에 이르고 있음.
            # 상기하는 바와 같이 동시설들은 인천공항에 대한 접근성을 확보하고 편의성을 증진하기 위한 목적으로 건설되었음에도 불구하고, 이용자들로 하여금 고가의 통행료를 부담토록 함으로써 접근의 편의성을 심각하게 저해함은 물론, MRG 등 협약을 통해 오히려 정부의 막대한 재정부담마저 초래하는 불합리한 상황에 봉착하고 있음.
            # 또한 동시설들은 공항과 연계된 사실상의 부속 교통시설로서 다변화하고 있는 공항산업과 항공산업의 글로벌 시장환경의 변화에 부응하여 동북아 허브공항으로서 인천국제공항의 서비스 경쟁력 제고에 긍정적으로 기여해야 함에도 불구하고, 상기와 같은 이유로 그 효과를 기대하기 어려운 현실에 놓이고 있음.
            # 이에 인천국제공항의 지속적인 글로벌 경쟁우위를 확보하고, 이용자들로 하여금 공항에 대한 접근편의성을 증진하고자 하는 목적으로, 인천국제공항 운영수익금의 일부를 투입하여 인천국제공항고속도로와 인천대교의 자금재조달 등 사업재구조화에 활용하고자 함.
            
            # 주요내용
            
            # 공사로 하여금 운영수익금의 일부를 인천공항 주변 민간투자 교통시설의 이용료 인하 등 인천공항의 경쟁력 제고를 위한 사업에 사용하도록 하는 근거를 신설함.""",
            """중학교 때 수학 전교 1등 - 초등학생 및 중학생 대상 수학 멘토로 활동. 청심국제고등학교 재학 당시 수학 및 통계 과목에서 매번 교과우수상 수상. 수학에서 (특히 초등학교 수학) 가장 중요한 것은 1) 개념 정리 및 원리 이해, 2) 특정 개념이 중학교 수학과 어떻게 연관되어 이어지고 심화되는지 이해하기, 3) 오답에 대해서는 틀린 이유와 원리를 이해하고 반복하여 풂으로써 자연스레 응용된 문제도 풀 수 있는 ‘생각하는 능력’ 향상하기라고 생각합니다. 수학 유형을 암기하기만 한다면 중고등학교 응용 수학에서 어려움이 생길 수밖에 없습니다. 그래서 저는 크게 두 가지에 집중할 예정입니다. 첫 번째는 수학적 개념/원리를 (중학교 교과와도 어떻게 이어지고 응용되는지를 포함) 이해하기 쉽게 정리된 노트로 만들어 학생들과 공유할 예정입니다. 두 번째로는, 오답 복습인데요, 아이가 오답 노트를 만들도록 하여 틀린 이유와 틀린 문제 뒤에 숨겨진 원리를 짧게라도 스스로 적어보는 시간을 갖는 것입니다. 정말 틀린 이유를 완벽하게 소화했는지 알기 위해서는 ‘같은 유형, 다른 문제’를 풀어보는 것이라고 생각하기 때문에 그 다음 수업 때 비슷한 유형의 또다른 문제를 반복적으로 아이에게 제공할 계획입니다! 4) 매주 복습 테스트를 진행하여 아이가 약한 부분이 무엇인지 파악할 예정입니다:)"""
        ]

        postproc_config = KeyPhrasePostProcConfig(

            # 절대 나올 필요가 없는 키워드
            drop_tokens=[
                '당시', '대상', '때', '뒤', '어려움', '다음', '1)', '4)', '학교', '데', '전', '무엇', '자신', '아이',
                '중학교', '초등학교', '중고등학교', '시절', '초등학생', '중학생',
                '테스트', '학년', '정도', '연령대', '스스로', '특정'
            ],
            drop_phrases=[],
            reject_candidate_keyphrase_pos_patterns=[],
            reject_candidate_keyphrase_str_patterns=['^\\d+점$', '^\\d+$', '^\\d\\)', '^등 ', '.+\\s등', '당시', '2019년 저', '년 저', '^점 ', '^\\d+년$'],
            allow_pos_patterns=['N', 'SN', 'SL'],
            allow_token_pos_pair_patterns=[]
        )
        predictor_config = PredictorConfig.parse_obj(self.predictor_config_json)
        config = KeyPhraseExtractorConfig(
            # methods=['mss', 'mmr02','mmr05', 'mmr07'],
            methods=['mss', 'mmr05'],
            post_proc_config=postproc_config,
            predictor_config=predictor_config
        )
        paani = KeyBert(config)

        for text in texts:
            final_ranking_res, rankers_res = paani.call_v2(text, refinery=None)
            print('=======================')
            print(text)
            print("detail_ranked_results:")
            print(rankers_res.pretty())
            print("ranked_keywords:")
            print(final_ranking_res.pretty())
            print('=======================')
            assert final_ranking_res is not None
