import logging
from typing import List, Dict
import json

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, SequenceLabelField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class SRLCopyNetDatasetReader(DatasetReader):

    def __init__(self,
                 target_namespace: str,
                 available_languages: Dict[str, int] = None,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._available_languages = available_languages or {"<EN>": 0,
                                                             "<EN-SRL>": 1,
                                                             "<DE>": 2,
                                                             "<DE-SRL>": 3,
                                                             "<FR>": 4,
                                                             "<FR-SRL>": 5}
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }

  
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = json.loads(line)
                if not line: continue
                yield self.text_to_instance("seq_words", "seq_tag_tokens", line)


    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

 
    def text_to_instance(self, source_key: str, target_key: str = None, line_obj: Dict = {}) -> Instance:
        

        # Read source and target
        target_sequence = line_obj.get(target_key, None)
        lang_src_token = line_obj["src_lang"].upper()
        lang_tgt_token = line_obj["tgt_lang"].upper()

        # Read Predicate Indicator and make Array
        verb_label = [0, 0] + [1 if label[-2:] == "-V" else 0 for label in line_obj["BIO"]] + [0]

        # Read Language Indicator and make Array
        lang_src_ix = self._available_languages[lang_src_token]
        lang_tgt_ix = self._available_languages[lang_tgt_token]
        # This array goes to the encoder as a whole
        lang_src_ix_arr = [0, 0] + [lang_src_ix for tok in line_obj[source_key]] + [0]
        # This array goes to each one of the decoder_steps
        lang_tgt_ix_arr = lang_tgt_ix # is just int for step decoder dimensionality

        # Tokenize Source
        tokenized_source = list(map(Token, line_obj[source_key])) # Data comes already tokenized!
        tokenized_source.insert(0, Token(lang_tgt_token))
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        # Process Target info during training...
        if target_sequence is not None:
            tokenized_target = list(map(Token, line_obj[target_key]))
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        # Add Verb Indicator to the Fields
        fields_dict['verb_indicator'] = SequenceLabelField(verb_label, source_field)
        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokenized_source[verb_label.index(1)].text
        meta_fields["verb"] = verb

        # Add Language Indicator to the Fields
        meta_fields["src_lang"] = lang_src_token
        meta_fields["tgt_lang"] = lang_tgt_token
        meta_fields["original_BIO"] = line_obj.get("BIO", [])
        meta_fields["original_predicate_senses"] = line_obj.get("pred_sense_origin", [])
        meta_fields["predicate_senses"] = line_obj.get("pred_sense", [])
        meta_fields["original_target"] = line_obj.get("seq_tag_tokens", [])
        fields_dict['language_enc_indicator'] = ArrayField(np.array(lang_src_ix_arr))
        fields_dict['language_dec_indicator'] = ArrayField(np.array(lang_tgt_ix_arr))

        fields_dict["metadata"] = MetadataField(meta_fields)
        return Instance(fields_dict)
    


input_directory = "./UP_English"

dataset_reader = SRLCopyNetDatasetReader(target_namespace="we-translate")

instances = []

for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(input_directory, filename)
        
        file_instances = list(dataset_reader.read(file_path))
        
        instances.extend(file_instances)


