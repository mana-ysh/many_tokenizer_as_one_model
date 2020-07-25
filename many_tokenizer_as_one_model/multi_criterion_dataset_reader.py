
import json
import os
from typing import Iterator, List, Dict, Tuple

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from many_tokenizer_as_one_model.tag_scheme import get_token_bio_from_strs


@DatasetReader.register('multi_criterion_dataset')
class MultiCriterionDatasetReader(DatasetReader):
    """
    file_path is just proxy to multiple criterion files.
    Assuming each file's formats like follows
    e.g) 私 は 蛙
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, charactors: List[Token], criterion_types: List[str], tags: List[str] = None) -> Instance:
        sentence_field = TextField(charactors, self.token_indexers)
        fields = {"sentence": sentence_field}
        fields["criterion_types"] = SequenceLabelField(
            labels=criterion_types,
            sequence_field=sentence_field,
            label_namespace="criterion_type"
        )
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field, label_namespace="labels")
            fields["labels"] = label_field

        return Instance(fields)

    def _parse_line(self, line: str) -> Tuple[List[Token], List[str]]:
        token_strs = line.rstrip().split(" ")
        tags = get_token_bio_from_strs(token_strs)
        charactors = [Token(char) for token_str in token_strs for char in token_str]
        assert len(charactors) == len(tags)
        return charactors, tags

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            dataset_config = json.load(f)

        dir_path = os.path.dirname(file_path)
        for critetion_type, file_name in dataset_config.items():
            _file_path = os.path.join(dir_path, file_name)
            with open(_file_path) as f:
                for line in f:
                    chars, tags = self._parse_line(line)
                    critetion_types = [critetion_type for _ in range(len(chars))]
                    yield self.text_to_instance(chars, critetion_types, tags)
