from typing import List, Dict
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Token
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
import numpy as np

from many_tokenizer_as_one_model.tag_scheme import tokenize_from_bios


class TokenizePredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, sentence: str, criterion_type: str) -> JsonDict:
        result = self.predict_json({"sentence": sentence, "criterion_type": criterion_type})
        tag_ids = np.argmax(result["tag_logits"], axis=-1)
        return tokenize_from_bios(sentence, [self._model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        chars = [Token(char) for char in sentence]
        criterion_types = [json_dict["criterion_type"] for _ in range(len(chars))]
        return self._dataset_reader.text_to_instance(chars, criterion_types)
