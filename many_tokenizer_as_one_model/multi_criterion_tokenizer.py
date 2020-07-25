
from typing import Dict

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
import torch


@Model.register('multi_criterion_tokenizer')
class MultiCriterionTokenizer(Model):
    def __init__(self,
                 char_embeddings: TextFieldEmbedder,
                 criterion_embeddings: Embedding,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.char_embeddings = char_embeddings
        self.criterion_embeddings = criterion_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                criterion_types: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        char_embeddings = self.char_embeddings(sentence)
        criterion_embeddings = self.criterion_embeddings(criterion_types)
        embeddings = char_embeddings + criterion_embeddings
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
