import logging
from typing import Dict, Tuple, List, Any, Union

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch

logger = logging.getLogger(__name__)  

class SRLCopyNetSeq2Seq(Model):
    

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 beam_size: int,
                 max_decoding_steps: int,
                 binary_pred_feature_dim: int = 0,
                 language_flag_dim: int = 0,
                 number_of_languages: int = 2,
                 target_embedding_dim: int = 100,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens",
                 tensor_based_metric: Metric = None,
                 token_based_metric: Metric = None
                 ) -> None:
        super().__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)  
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  
        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._target_namespace)

        self._tensor_based_metric = tensor_based_metric or \
            BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._token_based_metric = token_based_metric

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        
        if binary_pred_feature_dim > 0:
            self._binary_feature_embedding = Embedding(2, binary_pred_feature_dim)
        else:
            self._binary_feature_embedding = None

        
        if language_flag_dim > 0:
            self._language_embedding = Embedding(number_of_languages, language_flag_dim)
        else:
            self._language_embedding = None

        
        self._source_embedder = source_embedder
        self._encoder = encoder

        
        
        
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        
        
        
        
        
        
        
        self._target_embedder = Embedding(target_vocab_size, target_embedding_dim)
        self._attention = attention
        self._input_projection_layer = Linear(
                target_embedding_dim + language_flag_dim + self.encoder_output_dim * 2,
                self.decoder_input_dim)

        self._language_dec_indicator = None
        self._beam_size = beam_size

        
        
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        
        
        self._output_generation_layer = Linear(self.decoder_output_dim, target_vocab_size)

        
        
        
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        
        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

    @overrides
    def forward(self,  
                source_tokens: Dict[str, torch.LongTensor],
                source_token_ids: torch.Tensor,
                source_to_target: torch.Tensor,
                verb_indicator: torch.LongTensor,
                language_enc_indicator: torch.LongTensor,
                language_dec_indicator: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                target_tokens: Dict[str, torch.LongTensor] = None,
                target_token_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        
        self._language_dec_indicator = language_dec_indicator
        state = self._encode(source_tokens, verb_indicator, language_enc_indicator)
        state["source_token_ids"] = source_token_ids
        state["source_to_target"] = source_to_target

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loss(target_tokens, target_token_ids, self._language_dec_indicator, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                if self._tensor_based_metric is not None:
                    
                    top_k_predictions = output_dict["predictions"]
                    
                    best_predictions = top_k_predictions[:, 0, :]
                    
                    gold_tokens = self._gather_extended_gold_tokens(target_tokens["tokens"],
                                                                    source_token_ids,
                                                                    target_token_ids)
                    self._tensor_based_metric(best_predictions, gold_tokens)  
                if self._token_based_metric is not None:
                    predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                                  metadata,
                                                                  n_best=1)
                    self._token_based_metric(predicted_tokens,  
                                             [x["target_tokens"] for x in metadata])

        return output_dict

    
    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        batch_size, _ = state["source_mask"].size()

        
        
        
        final_encoder_output = util.get_final_encoder_states(
                state["encoder_outputs"],
                state["source_mask"],
                self._encoder.is_bidirectional())
        
        state["decoder_hidden"] = final_encoder_output
        
        state["decoder_context"] = state["encoder_outputs"].new_zeros(batch_size, self.decoder_output_dim)

        return state

    def _encode(self, source_tokens: Dict[str, torch.Tensor], verb_indicator: torch.LongTensor, lang_indicator: torch.LongTensor) -> Dict[str, torch.Tensor]:
        
        
        embedded_input = self._source_embedder(source_tokens)
        if self._binary_feature_embedding:
            embedded_verb_indicator = self._binary_feature_embedding(verb_indicator.long())
            embedded_input = torch.cat([embedded_input, embedded_verb_indicator], -1)
        if self._language_embedding:
            embedded_lang_indicator = self._language_embedding(lang_indicator.long())
            
            embedded_input = torch.cat([embedded_input, embedded_lang_indicator], -1)
        
        source_mask = util.get_text_field_mask(source_tokens)
        
        encoder_outputs = self._encoder(embedded_input, source_mask)
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      lang_indicator: torch.LongTensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        encoder_outputs_mask = state["source_mask"].float()
        
        embedded_input = self._target_embedder(last_predictions)

        if self._language_embedding:
            embedded_lang_indicator = self._language_embedding(lang_indicator.long())
            
            if len(embedded_lang_indicator.size()) == 3:
                if embedded_lang_indicator.size(1) == embedded_input.size(0):
                    embedded_lang_indicator = embedded_lang_indicator[0]
                else:
                    embedded_lang_indicator = embedded_lang_indicator.view(embedded_input.size(0), -1)
            embedded_input = torch.cat([embedded_input, embedded_lang_indicator], -1)
        
        
        attentive_weights = self._attention(
                state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)
        
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        
        selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)
        
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
                projected_decoder_input,
                (state["decoder_hidden"], state["decoder_context"]))
        return state

    
    def _forward_loss(self,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor,
                      language_indicator: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        batch_size, target_sequence_length = target_tokens["tokens"].size()

        
        source_mask = state["source_mask"]

        
        
        num_decoding_steps = target_sequence_length - 1
        
        
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        
        copy_mask = source_mask[:, 1:-1].float()
        
        
        
        
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())

        
        
        target_to_source = state["source_token_ids"].new_zeros(copy_mask.size())

        
        
        generation_scores_mask = state["decoder_hidden"].new_full((batch_size, self._target_vocab_size),
                                                                  fill_value=1.0)

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            
            input_choices = target_tokens["tokens"][:, timestep]
            
            
            
            if timestep < num_decoding_steps - 1:
                
                
                copied = ((input_choices == self._oov_index) &
                          (target_to_source.sum(-1) > 0)).long()
                
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                
                target_to_source = state["source_token_ids"] == target_token_ids[:, timestep+1].unsqueeze(-1)
            
            state = self._decoder_step(input_choices, selective_weights, language_indicator, state)
            
            
            generation_scores = self._get_generation_scores(state)
            
            
            
            copy_scores = self._get_copy_scores(state)
            
            step_target_tokens = target_tokens["tokens"][:, timestep + 1]
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                    generation_scores,
                    generation_scores_mask,
                    copy_scores,
                    step_target_tokens,
                    target_to_source,
                    copy_mask)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        
        
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        
        
        
        target_mask = util.get_text_field_mask(target_tokens)
        
        
        target_mask = target_mask[:, 1:].float()
        
        
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        
        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _get_input_and_selective_weights(self,
                                         last_predictions: torch.LongTensor,
                                         state: Dict[str, torch.Tensor]) -> Tuple[torch.LongTensor, torch.Tensor]:
        
        group_size, trimmed_source_length = state["source_to_target"].size()

        
        
        
        only_copied_mask = (last_predictions >= self._target_vocab_size).long()

        
        
        
        copy_input_choices = only_copied_mask.new_full((group_size,), fill_value=self._copy_index)
        input_choices = last_predictions * (1 - only_copied_mask) + copy_input_choices * only_copied_mask

        
        
        
        
        
        
        
        
        
        expanded_last_predictions = last_predictions.unsqueeze(-1).expand(group_size, trimmed_source_length)
        
        source_copied_and_generated = (state["source_to_target"] == expanded_last_predictions).long()

        
        
        
        
        
        
        adjusted_predictions = last_predictions - self._target_vocab_size
        
        
        adjusted_predictions = adjusted_predictions * only_copied_mask
        
        source_token_ids = state["source_token_ids"]
        
        adjusted_prediction_ids = source_token_ids.gather(-1, adjusted_predictions.unsqueeze(-1))
        
        
        
        source_only_copied = (source_token_ids == adjusted_prediction_ids).long()
        
        
        source_only_copied = source_only_copied * only_copied_mask.unsqueeze(-1)

        
        mask = source_only_copied | source_copied_and_generated
        
        selective_weights = util.masked_softmax(state["copy_log_probs"], mask)

        return input_choices, selective_weights

    
    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        _, trimmed_source_length = state["source_to_target"].size()

        
        
        
        
        
        
        input_choices, selective_weights = self._get_input_and_selective_weights(last_predictions, state)
        
        lang_dec_ind = torch.autograd.Variable(self._language_dec_indicator.data.repeat(self._beam_size, 1))
        state = self._decoder_step(input_choices, selective_weights, lang_dec_ind, state)
        
        
        generation_scores = self._get_generation_scores(state)
        
        
        
        copy_scores = self._get_copy_scores(state)
        
        
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        
        copy_mask = state["source_mask"][:, 1:-1].float()
        
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)
        
        
        log_probs = util.masked_log_softmax(all_scores, mask)
        
        generation_log_probs, copy_log_probs = log_probs.split(
                [self._target_vocab_size, trimmed_source_length], dim=-1)
        
        state["copy_log_probs"] = copy_log_probs
        
        
        
        
        
        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state

    def _get_predicted_tokens(self,
                              predicted_indices: Union[torch.Tensor, numpy.ndarray],
                              batch_metadata: List[Any],
                              n_best: int = None) -> List[Union[List[List[str]], List[str]]]:
        
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                if self._end_index in indices:
                    indices = indices[:indices.index(self._end_index)]
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._tensor_based_metric is not None:
                all_metrics.update(self._tensor_based_metric.get_metric(reset=reset))  
            if self._token_based_metric is not None:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))  
        return all_metrics
    
model = Seq2SeqCopyNet(Model)
