import torch
import sys
import warnings
sys.path.append('..')
warnings.filterwarnings('ignore')

from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import move_to_device
import json


import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np
from kodrop_electra.data_processing import BertDropTokenizer, BertDropReader, BertDropTokenIndexer
from kodrop_electra.augmented_koelectra import NumericallyAugmentedBERT
import argparse

parser = argparse.ArgumentParser(description='REST API for NAO with DRMC')

parser.add_argument("--path", "-p", default="/home/joon/Pyproject/ko_drop_electra/data/finance_dev.json", type=str, help="Dataset Path")
parser.add_argument("--model", "-m", default="/home/joon/Pyproject/ko_drop_electra/kdelectra_dir/best.th", type=str, help="Path for Model")
parser.add_argument("--device_num", "-n", default=2, type=int, help="CUDA device num")

config = parser.parse_args()



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class NumericallyAugmentedElectra:
    def __init__(self, config) -> None:
        self.device = torch.device('cuda:%d' % config.device_num)

        self.tokenizer = BertDropTokenizer('monologg/koelectra-base-discriminator')
        self.token_indexer = BertDropTokenIndexer('monologg/koelectra-base-discriminator')
        self.abert = NumericallyAugmentedBERT(Vocabulary(), 'monologg/koelectra-base-discriminator', special_numbers=[100, 1])
        self.reader = BertDropReader(self.tokenizer, {'tokens': self.token_indexer},extra_numbers=[0, 0])

        model = config.model

        model_dict = torch.load(model, map_location=self.device)
        abert_dict = self.abert.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if k in abert_dict}
        abert_dict.update(model_dict)
        self.abert.load_state_dict(abert_dict)
        self.abert.to(self.device)
        self.abert.eval()

    def read(self, passage, question):
        data ={  "finance_1": {
        "passage": passage,
        "qa_pairs": [
            {
                "question": question,
                "answer": {
                    "number": "1",
                    "date": {
                        "day": "",
                        "month": "",
                        "year": ""
                    },
                    "spans": []
                },
                "query_id": "f37e81fa-ef7b-4583-b671-762fc433faa9",
                "validated_answers": [
                    {
                        "number": "1",
                        "date": {
                            "day": "",
                            "month": "",
                            "year": ""
                        },
                        "spans": []
                    },
                    {
                        "number": "1",
                        "date": {
                            "day": "",
                            "month": "",
                            "year": ""
                        },
                        "spans": []
                    }
                ]
            },
        ]
    }
    }

        with open('sample.json', mode='w+') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)

        dev = self.reader.read('sample.json')
        iterator = BasicIterator(batch_size = 1)
        iterator.index_with(Vocabulary())

        dev_iter = iterator(dev, num_epochs=1)
        dev_batches = [batch for batch in dev_iter]
        self.dev_batches = move_to_device(dev_batches, config.device_num)

        # return self.dev_batches



    def infer(self):
        result = {}
        with torch.no_grad():
            for batch in self.dev_batches:
                output_dict = self.abert(**batch)
                for i in range(len(output_dict["question_id"])):
                    # result[output_dict["question_id"][i]] =  output_dict["answer"][i]["predicted_answer"]
                    result[output_dict["question_id"][i]] =  output_dict["answer"][i]['value']
        return result


    def query_sender(self, passage, question):
        self.read(passage, question)
        prediction = self.infer()

        return prediction

if __name__ == '__main__':
    passage = "FSN은 28일 개최한 임시주주총회에서 퓨쳐스트림네트웍스에서 FSN으로 상호변경 액면가 100원에서 500원으로 병합 등 주요 안건이 모두 승인됐다고 밝혔다. FSN은 내달 중 코스닥 상호 변경 상장이 진행될 예정이다."
    question =  "샤오미와 애플의 점유율 차이는 몇 % 포인트 인가?"

    na_electra = NumericallyAugmentedElectra(config)
    result = na_electra.query_sender(passage = passage, question=question)
    print(result)
