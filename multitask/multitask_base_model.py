from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import collections
import pandas as pd
import tensorflow as tf

from multitask import modules
import modules
from multitask import base_model
from constants import (RESULTS_CSV_FNAME,
                       MAX_CHECKPOINTS_TO_KEEP)

tf.logging.set_verbosity(tf.logging.INFO)


def _check_list_compatability(l, num_models):
    if not isinstance(l, (list, tuple)):
        raise TypeError("`l` %s must be list or tuple" % type(l))
    if len(l) != num_models:
        raise ValueError("l.len = %d != num_models %d" % (
            len(l), num_models))


def _check_fn_list_compatability(fn_list, num_models,
                                 assert_is_module=True):
    if len(fn_list) != num_models:
        raise ValueError("len(fn_list) %d != num_models %d" % (
            len(fn_list), num_models))
    for fn in fn_list:
        if assert_is_module and not isinstance(fn, modules.AbstractModule):
            raise TypeError(
                "Expected `fn_list` to be a subclass "
                "of AbstractModuleBaseEncoder, found ", type(fn))
        elif not callable(fn):
            raise TypeError("Expected `fn_list` to be callable")


def _mr_compatible(mixing_ratios, num_models, print_out=True):
    if len(mixing_ratios) != num_models:
        raise AssertionError(
            "mixing_ratios.len %d != num_models %d"
            % (len(mixing_ratios), num_models))
    if len(mixing_ratios) < 2:
        raise ValueError("Not supported")
    

    nz = 0
    for mr in mixing_ratios:
        if int(mr) != mr:
            raise ValueError("int(mr) != mr")
        if int(mr) < 1:
            nz += 1

    if nz >= len(mixing_ratios) - 1:
        raise ValueError("Not supported")

    if print_out:
        print("checked mixing_ratios = %s" % mixing_ratios)


def _write_results_to_csv(all_logits,
                          all_predictions,
                          all_fetched_data,
                          output_dir):

    results_df = pd.DataFrame(
        columns=["Logits", "Predictions", "Seq1", "Seq2", "Target"])
    results_df["Logits"] = all_logits
    results_df["Predictions"] = all_predictions
    # results_df["Seq1"] = all_fetched_data["seq_1"]
    # results_df["Seq2"] = all_fetched_data["seq_2"]
    # results_df["Target"] = all_fetched_data["target"]
    results_df.to_csv(output_dir)
    print("Wrote the results to %s" % output_dir)


class MultitaskBaseModel(base_model.BaseModel):
    def __init__(self,
                 names,
                 data,
                 embedding_fns,
                 encoder_fns_1,
                 encoder_fns_2,
                 logits_fns,
                 evaluation_fns,
                 # MTL
                 mixing_ratios,
                 L2_coefficient=None,
                 is_distill=False,
                 distill_coefficient_loc=None,
                 distill_coefficient_scale=None,
                 distill_temperature=1.0,
                 # optimization
                 optimizer="Adam",
                 learning_rate=0.001,
                 gradient_clipping_norm=2.0,
                 # misc
                 graph=None,
                 logdir=None,
                 main_model_index=0,
                 debug_mode=False):
        """
        Classification model that does the mapping of

            f: Seq_1 x Seq_2 --> Class
        """
        
        super(MultitaskBaseModel, self).__init__(
            logdir=logdir, graph=graph,
            saver_max_to_keep=MAX_CHECKPOINTS_TO_KEEP)

        num_models = len(names)
        _check_list_compatability(data, num_models)
        _check_fn_list_compatability(embedding_fns, num_models, True)
        _check_fn_list_compatability(encoder_fns_1, num_models, True)
        _check_fn_list_compatability(encoder_fns_2, num_models, True)
        _check_fn_list_compatability(logits_fns, num_models, False)
        _check_fn_list_compatability(evaluation_fns, num_models, False)

        # check mixing ratios and MTL
        if len(names) == 1:
            raise ValueError("Not supported")
        _mr_compatible(mixing_ratios, num_models, print_out=True)
        if main_model_index != 0:
            raise ValueError("`main_model_index` must be set to `0`")

        self._names = names
        self._data = data
        self._embedding_fns = embedding_fns
        self._encoder_fns_1 = encoder_fns_1
        self._encoder_fns_2 = encoder_fns_2
        self._logits_fns = logits_fns
        self._evaluation_fns = evaluation_fns

        # MTL
        self._mixing_ratios = mixing_ratios
        self._L2_coefficient = L2_coefficient
        self._is_disill = is_distill
        self._distill_temperature = distill_temperature
        self._distill_coefficient_loc = distill_coefficient_loc
        self._distill_coefficient_scale = distill_coefficient_scale

        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._gradient_clipping_norm = gradient_clipping_norm

        self._main_model_index = main_model_index
        self._debug = collections.defaultdict(list)
        self._debug_mode = debug_mode

    def _check_compatability(self):
        pass

    def _build(self):
        # build models
        (logits_collections,
         predictions_collections,
         step_collections,
         loss_collections,
         summary_collections,
         update_variables) = self._build_models()

        # build optimization and Ops
        train_op_collections = []
        summary_op_collections = []
        global_step_tensor = (  # common global step
            tf.train.get_or_create_global_step(graph=self._graph))

        for model_idx in range(self.num_models):

            loss = loss_collections[model_idx]
            summaries = summary_collections[model_idx]
            scope_name = "Opt_%s_%d" % (self._names[model_idx], model_idx)

            with tf.variable_scope(scope_name):
                train_op = self._build_optimizer(
                    loss, global_step_tensor,
                    update_variables=update_variables,
                    name=self._names[model_idx])

                summary_op = tf.summary.merge(
                    inputs=summaries,
                    name=self._names[model_idx])

                train_op_collections.append(train_op)
                summary_op_collections.append(summary_op)

        self._logits_collections = logits_collections
        self._predictions_collections = predictions_collections
        self._step_collections = step_collections
        self._loss_collections = loss_collections
        self._train_op_collections = train_op_collections
        self._summary_op_collections = summary_op_collections
        # a tensor
        self._global_step_tensor = global_step_tensor


    def _build_models(self):
        """building MultiTask Models"""
        raise NotImplementedError

    def _build_single_model(self,
                            task_name,
                            data,
                            embedding_fn,
                            encoder_fn_1,
                            encoder_fn_2,
                            logits_fn):

        """building Individual Models"""
        raise NotImplementedError

    def _build_optimizer(self,
                         loss,
                         global_step_tensor,
                         update_variables,
                         name):
        # Add the optimizer.
        # ------------------------------------------------------
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step_tensor,
                learning_rate=self._learning_rate,
                optimizer=self._optimizer,
                # some gradient clipping stabilizes training in the beginning.
                clip_gradients=self._gradient_clipping_norm,
                # name="OptimizeLoss_%s" % name,
                variables=update_variables,
                summaries=["loss", "learning_rate", "global_gradient_norm"])

        return train_op

    def initialize_data_iterator(self, model_idx=None):
        """
        Initialize data generators. This function assumes the
        wrapped model to have `._data` field.

        Args:
            model_idx: the index of the model to be initialized.
                       Set to None to initialize all models
        """
        if model_idx and not isinstance(model_idx, (list, tuple)):
            raise TypeError("`model_idx` must be list-like structure")

        model_indices = (
            # if model_idx is None
            # model_idx = [all_models_indices]
            model_idx if model_idx is not None
            else range(self.num_models))

        for idx in model_indices:
            try:
                print("Initializing %d data" % idx)
                self._sess.run(self._data[idx].initializer)
            except AttributeError:
                # wraps the error with more informative message
                raise AttributeError("Model #%d has no `_data` attribute")

    def _fetch_data_batch(self, logits, predictions, data=None):
        """Fetch Data and Predictions

        Args:
            predictions: predictions to fetch
            data: Data object to fetch, use None if not needed

        Returns:
            fetched_predictions:
                list: [batch size]
                outputs of model predictions

            fetched_data: fetched data
                tuple: [source_1, source_2, target]
                each of which is a list: [batch_size]
                input data that makes the predictions
        """
        if data is not None:
            # this will explicitly avoid data.initializer
            (fetched_target,
             fetched_logits,
             fetched_predictions) = self._sess.run(
                [data.target, logits, predictions])

            fetched_logits = fetched_logits.tolist()
            fetched_predictions = fetched_predictions.tolist()
            fetched_data = {"target": fetched_target.tolist()}

        else:
            fetched_data = None
            (fetched_logits,
             fetched_predictions) = self._sess.run(
                [logits, predictions])
            
            fetched_logits = fetched_logits.tolist()
            fetched_predictions = fetched_predictions.tolist()

        return fetched_logits, fetched_predictions, fetched_data

    def _evaluate(self,
                  logits,
                  predictions,
                  data, evaluation_fn,
                  max_eval_batches=None,
                  calculate_scores=True,
                  write_results=False):
        """Sample from model predictions, and evaluate outputs

        Args:
            logits:
                Tensor
                Logits to fetch
            predictions:
                Tensor
                predictions to fetch

            data:
                BatchedInput(Tensor)
                Data object to fetch

            evaluation_fn:
                Callable(predictions, source_1, source_2, target) --> R
                a function that evaluate the predictions

            write_results:
                Bool
                Whether to write results to csv file for diagnosis

        Returns:
            counts:
                Integer
                number of predictions

            scores:
                Float
                the score from the evaluation
        """
        # counting the evaluation batches
        num_eval_batches = 0
        # logits and predictions from the model
        all_logits = []
        all_predictions = []
        # fetched data that led to the predictions
        # dictionary of {seq_1: [], seq_2: [], target: []}
        all_fetched_data = collections.defaultdict(list)
        try:
            while True:
                # sample predictions
                (fetched_logits,
                 fetched_predictions,
                 fetched_data) = self._fetch_data_batch(
                    logits=logits, predictions=predictions, data=data)

                # Cache the data
                all_logits += fetched_logits
                all_predictions += fetched_predictions
                all_fetched_data["target"] += fetched_data["target"]

                # break the loop if max_eval_batches is set
                num_eval_batches += 1
                if (max_eval_batches and
                        num_eval_batches >= max_eval_batches):
                    break

        except tf.errors.OutOfRangeError:
            pass

        # Evaluate
        scores = None
        if calculate_scores:
            scores = evaluation_fn(
                all_predictions,
                all_fetched_data["seq_1"],  # Should be empty
                all_fetched_data["seq_2"],  # Should be empty
                all_fetched_data["target"])

        if write_results:
            _write_results_to_csv(
                all_logits,
                all_predictions,
                all_fetched_data,
                output_dir=os.path.join(
                    self._logdir, RESULTS_CSV_FNAME))

        return len(all_predictions), scores

    def _get_global_step(self):
        return self._step_collections["GlobalStep"]

    @property
    def num_models(self):
        return len(self._names)

    @property
    def num_tasks(self):
        return len(self._names)

    @property
    def not_multitask(self):
        return self.num_models == 1

    @property
    def total_steps(self):
        # the difference between GS and TS
        # is that GS will be restored from checkpoint
        # while TS will be re-initialized always
        task_steps = [val for key, val in self._step_collections.items()
                      if key != "GlobalStep"]
        return sum(task_steps)

    @property
    def main_task_step(self):
        main_task_name = self._names[self._main_model_index]
        return self._step_collections[main_task_name]

    def _task_selector(self, step):
        if self._mixing_ratios is None:
            return self._main_model_index

        # e.g. self._mixing_ratios = [2, 1, 5]
        # then this will flatten the list into
        # [0, 0, 1, 2, 2, 2, 2, 2]
        flatten_mixing_ratios = [idx for idx, ratio in enumerate(
            self._mixing_ratios) for _ in range(ratio)]

        # suppose the step is 1001 we first do
        # step mod sum(ratios), and the remainder is the extra steps
        # then flatten_mixing_ratios[remainder] gives us the next index
        remainder = step % sum(self._mixing_ratios)
        task = flatten_mixing_ratios[remainder]

        return task

    def train(self, model_idx=None, print_message=False):
        model_idx = model_idx if model_idx is not None else self._task_selector(self.global_step)
        model_name = "%s-%d" % (self._names[model_idx], model_idx)

        # TRAIN ONE STEP
        # ------------------------------------------
        fetches = {
            "GlobalStep": self._global_step_tensor,
            "Loss": self._loss_collections[model_idx],
            "TrainOp": self._train_op_collections[model_idx]}

        fetched = self._sess.run(fetches=fetches)
        loss = fetched["Loss"]
        global_step = fetched["GlobalStep"]

        # Update Statistics
        # ------------------------------------------
        # update steps
        self._step_collections[model_name] += 1
        self._step_collections["GlobalStep"] = global_step

        # and print info
        message = self._format_message()

        return loss, message

    def evaluate(self, model_idx,
                 max_eval_batches=None,
                 write_results=False,
                 write_to_summary=True):
        """Sample from model predictions, and evaluate outputs"""
        
        # write_summary needs global_step
        self._step_collections["GlobalStep"] = (
            self._sess.run(self._global_step_tensor))

        # evaluate the model
        model_name = "%s-%d" % (self._names[model_idx], model_idx)
        counts, scores = self._evaluate(
            data=self._data[model_idx],
            logits=self._logits_collections[model_idx],
            predictions=self._predictions_collections[model_idx],
            evaluation_fn=self._evaluation_fns[model_idx],
            max_eval_batches=max_eval_batches,
            calculate_scores=True,
            write_results=write_results)

        # write summary
        if write_to_summary:
            self.write_summary(
                "TASK-%s/ValScores" % model_name, scores)

        return {"MAIN": scores}

    def inference(self, model_idx):
        return self._evaluate(
            data=self._data[model_idx],
            logits=self._logits_collections[model_idx],
            predictions=self._predictions_collections[model_idx],
            evaluation_fn=self._evaluation_fns[model_idx],
            max_eval_batches=None,
            calculate_scores=False,
            write_results=True)


    def _format_message(self):
        # print step information
        message = "#%d" % self.global_step
        return message



class MultitaskSingleAndDualStreamBaseModel(MultitaskBaseModel):
    
    def _build_single_model(self,
                            task_name,
                            data,
                            embedding_fn,
                            encoder_fn_1,
                            encoder_fn_2,
                            logits_fn):

        """building Individual Models"""
        if task_name in ["CoLA", "SST"]:
            _single_model_fn = self._build_single_stream_model
        else:
            _single_model_fn = self._build_dual_stream_model

        return _single_model_fn(
            data=data,
            embedding_fn=embedding_fn,
            encoder_fn_1=encoder_fn_1,
            encoder_fn_2=encoder_fn_2,
            logits_fn=logits_fn)

    def _build_single_stream_model(self,
                                   data,
                                   embedding_fn,
                                   encoder_fn_1,
                                   encoder_fn_2,
                                   logits_fn):
        """BiLSTM with max pooling, but ignore sequence_2"""
        
        # Build the Model
        # ------------------------------------------------------
        
        # Embedding Function:
        # `modules.TFHubElmoEmbedding` requires additional
        # sequence_length for masking, while `modules.Embedding` does not
        if isinstance(embedding_fn, (modules.Embeddding,
                                     modules.CachedElmoModule)):
            embedded_tokens_1 = embedding_fn(data.source_1)

        if isinstance(embedding_fn, modules.TFHubElmoEmbedding):
            embedded_tokens_1 = embedding_fn(
                data.source_1, data.source_1_sequence_length)

        # Encoder Function:
        # outputs: [batch_size, length, num_units x 2]

        if isinstance(encoder_fn_1, modules.LstmEncoder):
            outputs_1, _ = encoder_fn_1(
                inputs=embedded_tokens_1,
                sequence_length=data.source_1_sequence_length)

        if isinstance(encoder_fn_1, modules.TransformerEncoder):
            outputs_1 = encoder_fn_1(
                inputs=embedded_tokens_1,
                sequence_length=data.source_1_sequence_length)

        if isinstance(encoder_fn_1, modules.PairEncoderWithAttention):
            raise ValueError("In single stream model, "
                             "`PairEncoderWithAttention` is not supported")
            # outputs_1, outputs_2 = encoder_fn_1(
            #     sequence_1=embedded_tokens_1,
            #     sequence_2=embedded_tokens_2,
            #     sequence_1_length=data.source_1_sequence_length,
            #     sequence_2_length=data.source_2_sequence_length)

        # processed_outputs: [batch_size, num_units x 2]
        features = tf.reduce_max(outputs_1, axis=1)
        # final linear layer
        logits = logits_fn(features)

        # Compute current predictions and cross entropies
        # ------------------------------------------------------
        # [batch_size, num_classes]
        predictions = tf.argmax(logits, axis=1)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=data.target, logits=logits))

        # For Unit Tests
        # -------------------------------------------------------
        if self._debug_mode:
            def _add_to_debug(var_name, var):
                self._debug[var_name].append(var)
            _add_to_debug("embedded_tokens_1", embedded_tokens_1)
            _add_to_debug("outputs_1", outputs_1)
            _add_to_debug("features", features)
            _add_to_debug("logits", logits)
            _add_to_debug("predictions", predictions)
            _add_to_debug("cross_entropy", cross_entropy)

        return logits, predictions, cross_entropy


    def _build_dual_stream_model(self,
                                 data,
                                 embedding_fn,
                                 encoder_fn_1,
                                 encoder_fn_2,
                                 logits_fn):
        """BiLSTM with max pooling from
            https://arxiv.org/pdf/1705.02364.pdf"""
        # Build the Model
        # ------------------------------------------------------
        
        # Embedding Function:
        # `modules.TFHubElmoEmbedding` requires additional
        # sequence_length for masking, while `modules.Embedding` does not
        if isinstance(embedding_fn, (modules.Embeddding,
                                     modules.CachedElmoModule)):
            embedded_tokens_1 = embedding_fn(data.source_1)
            embedded_tokens_2 = embedding_fn(data.source_2)

        if isinstance(embedding_fn, modules.TFHubElmoEmbedding):
            embedded_tokens_1 = embedding_fn(
                data.source_1, data.source_1_sequence_length)
            embedded_tokens_2 = embedding_fn(
                data.source_2, data.source_2_sequence_length)

        # Encoder Function:
        # (`encoder_fn_1` and `encoder_fn_2` will come from the same class)
        # `modules.LstmEncoder` returns cell_outputs and last_cell_state
        # and we only need one
        # `modules.TransformerEncoder` returns only combined processed sequence
        # `modules.PairEncoderWithAttention` returns processed sequences
        # for both sequence inputs, and we don't need `encoder_fn_2`

        # Note that:
        # outputs: [batch_size, length, num_units x 2]
        # this doesn't allow layer-sharing, but will be supported
        # in the future

        if isinstance(encoder_fn_1, modules.LstmEncoder):
            outputs_1, _ = encoder_fn_1(
                inputs=embedded_tokens_1,
                sequence_length=data.source_1_sequence_length)

            outputs_2, _ = encoder_fn_2(
                inputs=embedded_tokens_2,
                sequence_length=data.source_2_sequence_length)

        if isinstance(encoder_fn_1, modules.TransformerEncoder):
            outputs_1 = encoder_fn_1(
                inputs=embedded_tokens_1,
                sequence_length=data.source_1_sequence_length)

            outputs_2 = encoder_fn_2(
                inputs=embedded_tokens_2,
                sequence_length=data.source_2_sequence_length)

        if isinstance(encoder_fn_1, modules.PairEncoderWithAttention):
            outputs_1, outputs_2 = encoder_fn_1(
                sequence_1=embedded_tokens_1,
                sequence_2=embedded_tokens_2,
                sequence_1_length=data.source_1_sequence_length,
                sequence_2_length=data.source_2_sequence_length)

        # InferSent-style Pooling and Classifier
        # row-level max-pooling
        # processed_outputs: [batch_size, num_units x 2]
        u = tf.reduce_max(outputs_1, axis=1)
        v = tf.reduce_max(outputs_2, axis=1)

        # Using [u, v, |u - v|, u * v] as features
        u_mul_v = tf.multiply(u, v)
        u_min_v = tf.abs(tf.subtract(u, v))
        features = tf.concat([u, v, u_min_v, u_mul_v], axis=-1)

        # final linear layer
        logits = logits_fn(features)

        # Compute current predictions and cross entropies
        # ------------------------------------------------------
        # [batch_size, num_classes]
        predictions = tf.argmax(logits, axis=1)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=data.target, logits=logits))

        # For Unit Tests
        # -------------------------------------------------------
        if self._debug_mode:
            def _add_to_debug(var_name, var):
                self._debug[var_name].append(var)
            _add_to_debug("embedded_tokens_1", embedded_tokens_1)
            _add_to_debug("embedded_tokens_2", embedded_tokens_2)
            _add_to_debug("outputs_1", outputs_1)
            _add_to_debug("outputs_2", outputs_2)
            _add_to_debug("u", u)
            _add_to_debug("v", v)
            _add_to_debug("u_mul_v", u_mul_v)
            _add_to_debug("u_min_v", u_min_v)
            _add_to_debug("features", features)
            _add_to_debug("logits", logits)
            _add_to_debug("predictions", predictions)
            _add_to_debug("cross_entropy", cross_entropy)

        return logits, predictions, cross_entropy
