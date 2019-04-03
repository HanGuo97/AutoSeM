from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from utils import misc_utils
from data_utils import vocab_utils
from data_utils import iterator_utils_3
from utils import merge_vocabs
import modules
from multitask import modules

from multitask import multitask_models
from constants import (MAIN_MODEL_INDEX,
                       NUM_TASKS,
                       TRANSFORMER_NUM_HEADS,
                       CACHED_ELMO_NUM_ELEMENTS,
                       CACHED_ELMO_NUM_UNITS,
                       DATA_BUFFER_MULTIPLIER,
                       DATA_NUM_PARALLEL_CALLS,
                       USING_ELMO)


JOINT_VOCAB_FILE = "joint_source_vocab"


def _merge_vocabs(hparams):
    vocab_files = [f + ".source_vocab" for f in hparams.train_files]
    merged_vocab_file = os.path.join(hparams.logdir, JOINT_VOCAB_FILE)
    special_tokens = [vocab_utils.EOS, vocab_utils.SOS, vocab_utils.UNK]

    merge_vocabs.merge_vocabs(
        names=hparams.task_names,
        vocab_files=vocab_files,
        joint_vocab_file=merged_vocab_file,
        build_indices=False,
        special_tokens=special_tokens)

    return merged_vocab_file


def _build_data(train_file, val_file, src_vocab_file,
                train_batch_size, val_batch_size,
                train_graph, val_graph, random_seed):
    
    # iterator_utils_2 return ELMO embeddings
    iterator_builder = (
        iterator_utils_3.get_pairwise_classification_iterator)
    # Note that the label.vocab file for train/val/test
    # can be different, i.e., the same label will be mapped
    # to a differnt integer with different label.vocab file
    # so ALWAYS use train.label_vocab for consistency
    tgt_vocab_file = train_file + ".label_vocab"

    (token_vocab_size,
     src_vocab_file) = vocab_utils.check_vocab(
        vocab_file=src_vocab_file,
        out_dir=os.path.dirname(src_vocab_file),
        check_special_token=True)

    (label_vocab_size,
     tgt_vocab_file) = vocab_utils.check_vocab(
        vocab_file=tgt_vocab_file,
        out_dir=os.path.dirname(tgt_vocab_file),
        check_special_token=False)

    tf.logging.info("token_vocab_size = %d from %s" % (
        token_vocab_size, src_vocab_file))
    tf.logging.info("label_vocab_size = %d from %s" % (
        label_vocab_size, tgt_vocab_file))

    def _data_generator(fname):
        h5py_data = h5py.File(fname + ".elmo.hdf5", "r")
        # Iterating over `.keys()` and find length or
        # using `len(h5py_data)` in large datasets
        # prohibitative, instead, `sentence_to_index`
        # is much faster to get. We can also use
        # while-loop, and break when output is None.
        # Speed: When dataset is small (e.g. RTE/MRPC)
        # using `len()` is roughly 10-20X faster than my
        # approach (40ms vs. 2ms). When dataset is large (e.g. QNLI)
        # my approach takes 500ms versus 2min for `len()`.
        sentence_to_index = eval(  # "{S1: Index1, S2: Index2,...}"
            h5py_data.get("sentence_to_index").value[0])
        # Instead of using `len(sentence_to_index.keys())`
        # use `max(sentence_to_index.values)` because there
        # can be duplicate sentences, and thus using `len()`
        # will lead to smaller `num_elements` than supposed to be.
        # `+1` because max index + 1 = length
        # Note that this might also lead to incorrect count
        # because in duplicate settings, we cannot guarantee
        # that `max(sentence_to_index.values)` will return
        # the correct length when the multiple sentences
        # are duplicate form of the sentence at the max index.
        # But this might not be a problem here because the max index
        # is usually inserted at the end, and thus will not be
        # overriden. Some tests will be used to verify this
        num_elements = np.max([int(i) for i in sentence_to_index.values()]) + 1

        def _callable_generator():
            for i in range(num_elements):
                # [3, sequence_length, 1024]
                raw_data = h5py_data.get(str(i))
                raw_data = raw_data.value
                yield raw_data
        return _callable_generator

    # train dataset
    with train_graph.as_default():
        # no UNKs in target labels
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

        train_src_1 = tf.data.Dataset.from_generator(
            _data_generator(train_file + ".sequence_1"),
            output_types=tf.float32,
            output_shapes=tf.TensorShape(
                [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
        train_src_2 = tf.data.Dataset.from_generator(
            _data_generator(train_file + ".sequence_2"),
            output_types=tf.float32,
            output_shapes=tf.TensorShape(
                [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
        train_tgt = tf.data.TextLineDataset(train_file + ".labels")
        train_batch = iterator_builder(
            src_dataset_1=train_src_1,
            src_dataset_2=train_src_2,
            tgt_dataset=train_tgt,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=train_batch_size,
            random_seed=random_seed,
            src_len_axis=1,
            num_parallel_calls=DATA_NUM_PARALLEL_CALLS,
            output_buffer_size=train_batch_size * DATA_BUFFER_MULTIPLIER,
            shuffle=True,
            repeat=True)

    # val dataset
    with val_graph.as_default():
        # since these are graph-specific, we build them twice
        # no UNKs in target labels
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file)

        val_src_1 = tf.data.Dataset.from_generator(
            _data_generator(val_file + ".sequence_1"),
            output_types=tf.float32,
            output_shapes=tf.TensorShape(
                [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
        val_src_2 = tf.data.Dataset.from_generator(
            _data_generator(val_file + ".sequence_2"),
            output_types=tf.float32,
            output_shapes=tf.TensorShape(
                [CACHED_ELMO_NUM_ELEMENTS, None, CACHED_ELMO_NUM_UNITS]))
        val_tgt = tf.data.TextLineDataset(val_file + ".labels")
        val_batch = iterator_builder(
            src_dataset_1=val_src_1,
            src_dataset_2=val_src_2,
            tgt_dataset=val_tgt,
            tgt_vocab_table=tgt_vocab_table,
            batch_size=val_batch_size,
            random_seed=random_seed,
            src_len_axis=1,
            num_parallel_calls=DATA_NUM_PARALLEL_CALLS,
            output_buffer_size=val_batch_size * DATA_BUFFER_MULTIPLIER,
            shuffle=False,
            repeat=False)

    return train_batch, val_batch, token_vocab_size, label_vocab_size


def _build_model(hparams,
                 data_batches,
                 num_classes,
                 vocab_size,
                 # misc
                 graph,
                 is_training,
                 debug_mode=False):

    # ModelTypes
    # -----------------------------------
    additional_kwargs = {}
    wrapper_kwargs = {}


    # determine the base type of the model
    if hparams.multitask_model_type is None:
        raise ValueError("`multitask_model_type` can't be None")

    elif hparams.multitask_model_type == "Hard":
        ModelCreator = multitask_models.MultitaskHardSharingModel

    # determine the auto-parts
    if hparams.auto_model_type == "AutoMR":
        ModelWrapper = multitask_models.MTLAutoMRModel
        wrapper_kwargs = {
            "initial_weight": 100.0,
            "update_rate": hparams.automr_update_rate,
            "reward_scale": hparams.automr_reward_scale,
            "temperature_anneal_rate": None}

    print(misc_utils.bcolors.WARNING +
          "Using Model %s" % (ModelCreator.__base__) +
          misc_utils.bcolors.ENDC)

    # Create Parameter Sharing Rules
    # -----------------------------------
    with graph.as_default():
        (embedding_fns,
         encoder_fns_1,
         encoder_fns_2,
         logits_fns,
         evaluation_fns) = base_functions(
            hparams=hparams,
            num_classes=num_classes,
            vocab_size=vocab_size,
            is_training=is_training)

    model = ModelCreator(
        names=hparams.task_names,
        data=data_batches,
        embedding_fns=embedding_fns,
        encoder_fns_1=encoder_fns_1,
        encoder_fns_2=encoder_fns_2,
        logits_fns=logits_fns,
        evaluation_fns=evaluation_fns,
        # MTL
        mixing_ratios=hparams.mixing_ratios,
        # optimization
        optimizer="Adam",
        learning_rate=hparams.learning_rate,
        gradient_clipping_norm=2.0,
        # misc
        graph=graph,
        logdir=hparams.logdir,
        main_model_index=MAIN_MODEL_INDEX,
        debug_mode=debug_mode,
        # additional args
        **additional_kwargs)

    model = ModelWrapper(
        model=model,
        **wrapper_kwargs)

    model.build()
    return model


def build_model(hparams, debug_mode=False):
    # build the data
    train_batches = []
    val_batches = []
    token_vocab_sizes = []
    label_vocab_sizes = []
    train_graph = tf.Graph()
    val_graph = tf.Graph()

    merged_src_vocab_file = _merge_vocabs(hparams)
    hparams.merged_src_vocab_file = merged_src_vocab_file
    for train_file, eval_file in zip(hparams.train_files,
                                     hparams.eval_files):
        
        (train_batch,
         val_batch,
         token_vocab_size,
         label_vocab_size) = _build_data(
            train_file=train_file,
            val_file=eval_file,
            src_vocab_file=merged_src_vocab_file,
            train_batch_size=hparams.train_batch_size,
            val_batch_size=hparams.eval_batch_size,
            train_graph=train_graph,
            val_graph=val_graph,
            random_seed=hparams.tensorflow_seed)

        train_batches.append(train_batch)
        val_batches.append(val_batch)
        token_vocab_sizes.append(token_vocab_size)
        label_vocab_sizes.append(label_vocab_size)

    # they all must come from the same vocab
    # thus `token_vocab_size` can be directly used
    misc_utils.assert_all_same(token_vocab_sizes)

    # build the models
    # ------------------------------------------
    train_MTL_model = _build_model(
        hparams=hparams,
        data_batches=train_batches,
        num_classes=label_vocab_sizes,
        vocab_size=token_vocab_size,
        graph=train_graph,
        is_training=True,
        debug_mode=debug_mode)

    val_MTL_model = _build_model(
        hparams=hparams,
        data_batches=val_batches,
        num_classes=label_vocab_sizes,
        vocab_size=token_vocab_size,
        graph=val_graph,
        is_training=False,
        debug_mode=debug_mode)

    return train_MTL_model, val_MTL_model


def base_functions(hparams,
                   num_classes,
                   vocab_size,
                   is_training):

    num_models = len(hparams.tasks)
    if len(num_classes) != num_models:
        raise ValueError("len(num_classes) != num_models")

    if hparams.base_model_type not in ["LSTM"]:
        raise ValueError("hparams.base_model_type not in LSTM")

    if USING_ELMO != (hparams.embedding_type == "ELMO"):
        raise ValueError("This is Wrong")

    # EMBEDDING MODULES
    # ------------------------------------------------
    # Share All the embeddings
    if hparams.embedding_type == "CachedELMO":
        # The data is the computed ELMO representations
        # so no need for extra embedding function
        embedding_fn = modules.CachedElmoModule()

    if hparams.embedding_type == "ELMO":
        embedding_fn = modules.TFHubElmoEmbedding()
    
    if hparams.embedding_type == "RandInit":
        embedding_fn = modules.Embeddding(
            vocab_size=vocab_size,
            embed_dim=hparams.embedding_dim)

    # ENCODERS MODULES
    # ------------------------------------------------
    # Share two encoders
    if hparams.base_model_type == "LSTM":
        encoder_fns_1 = encoder_fns_2 = [
            modules.LstmEncoder(
                unit_type="lstm",
                num_units=hparams.num_units,
                num_layers=hparams.num_layers,
                dropout_rate=hparams.dropout_rate,
                is_training=is_training,
                name="LstmEncoder_%s_%d" % (name, tid))
            for tid, name in enumerate(hparams.task_names)]

    # PROJECTION LAYERS
    # ------------------------------------------------
    ProjLayers = [
        tf.layers.Dense(
            units=num_class,
            name="LogitsLayer_%s_%d" % (name, tid))
        for tid, (num_class, name) in enumerate(
            zip(num_classes, hparams.task_names))]

    return [
        # share all embedding layers
        [embedding_fn] * num_models,
        encoder_fns_1,
        encoder_fns_2,
        ProjLayers,
        [t.evaluate for t in hparams.tasks]]
