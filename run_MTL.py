from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import numpy as np
from tqdm import trange
import tensorflow as tf

import model_utils
import hparams as hps_utils
from multitask import tasks
from utils import misc_utils
from utils import training_manager
from multitask import multitask_models
from constants import (MAIN_MODEL_INDEX,
                       TRAIN_LOGFILE_SUFFIX,
                       INFER_LOGFILE_SUFFIX,
                       EARLY_STOP_TOLERANCE,
                       AUTOMR_MAX_EVAL_BATCHES)

tf.logging.set_verbosity(tf.logging.INFO)


# ==================================================
# Command Line Arguments
# ==================================================

def get_hparams():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--tasks",
                        type=str, default=None)
    parser.add_argument("--max_steps",
                        type=int, default=None)
    parser.add_argument("--steps_per_eval",
                        type=int, default=None)
    parser.add_argument("--logdir",
                        type=str, default=None)
    parser.add_argument("--ckpt_file",
                        type=str, default=None)
    parser.add_argument("--random_seed",
                        type=int, default=None)
    # Inference
    parser.add_argument("--infer",
                        action="store_true", default=False)

    # -----------------------------------------
    # HYPER-PARAMETERS

    # Model
    parser.add_argument("--embedding_dim",
                        type=int, default=512)
    parser.add_argument("--num_units",
                        type=int, default=512)
    parser.add_argument("--num_layers",
                        type=int, default=2)
    parser.add_argument("--dropout_rate",
                        type=float, default=0.5)
    parser.add_argument("--learning_rate",
                        type=float, default=0.001)

    # MTL
    parser.add_argument("--model_type",
                        type=str, default=None)
    parser.add_argument("--mixing_ratios",
                        type=str, default="AutoMR")
    parser.add_argument("--training_strategy",
                        type=str, default=None)

    # AutoMR
    parser.add_argument("--automr_update_rate",
                        type=float, default=0.3)
    parser.add_argument("--automr_reward_scale",
                        type=float, default=1.0)
    parser.add_argument("--stage",
                        type=int, default=None)

    # Specific MTL
    parser.add_argument("--is_distill",
                        action="store_true", default=False)
    parser.add_argument("--loss_coefficient_loc",
                        type=float, default=None)
    parser.add_argument("--loss_coefficient_scale",
                        type=float, default=None)
    parser.add_argument("--distill_temperature",
                        type=float, default=1.0)
    # -----------------------------------------
    FLAGS, unparsed = parser.parse_known_args()

    if unparsed:
        raise ValueError(unparsed)

    return _get_hparams(FLAGS)


def _get_hparams(FLAGS):
    # just being lazy
    ChosenTasks = (
        [tasks.problem(task) for task in FLAGS.tasks.split("-")]
        if FLAGS.tasks is not None else None)
    MainTask = ChosenTasks[MAIN_MODEL_INDEX]
    isAuto = (FLAGS.mixing_ratios.startswith("Auto")
              if FLAGS.mixing_ratios is not None else False)

    _logdir = (
        FLAGS.logdir if not FLAGS.logdir.endswith("/")
        else FLAGS.logdir[:-1])
    train_logfile = _logdir + TRAIN_LOGFILE_SUFFIX
    infer_logfile = _logdir + INFER_LOGFILE_SUFFIX

    print("\t\tRunning %d" % FLAGS.stage)
    if FLAGS.stage == 2:
        raise ValueError(
            "To run the stage-2, please follow the instructions"
            "on https://scikit-optimize.github.io. Other existing "
            "alternatives include: "
            "`https://github.com/fmfn/BayesianOptimization` "
            "`https://github.com/HIPS/Spearmint`.")

    """A set of basic hyperparameters."""
    return tf.contrib.training.HParams(
        # Tasks and data files
        # ---------------------------------
        tasks=ChosenTasks,
        task_names=[t.name for t in ChosenTasks],
        train_files=[t.train_data for t in ChosenTasks],
        eval_files=([t.val_data for t in ChosenTasks]
                    if not FLAGS.infer else
                    [t.test_data for t in ChosenTasks]),
        # Batch sizes
        # ---------------------------------
        # just using the main task info
        train_batch_size=MainTask.train_batch_size,
        eval_batch_size=MainTask.evaluate_batch_size,
        # Steps
        # ---------------------------------
        max_steps=(
            FLAGS.max_steps
            if FLAGS.max_steps is not None
            else MainTask.max_steps),
        steps_per_eval=(
            FLAGS.steps_per_eval
            if FLAGS.steps_per_eval is not None
            else MainTask.steps_per_eval),
        # Training
        # ---------------------------------
        logdir=FLAGS.logdir,
        manager_logdir=FLAGS.logdir,
        ckpt_file=FLAGS.ckpt_file,  # initialize model, or run test
        numpy_seed=FLAGS.random_seed,
        tensorflow_seed=FLAGS.random_seed,
        train_logfile=train_logfile,
        # Inference
        # ---------------------------------
        infer=FLAGS.infer,
        infer_logfile=infer_logfile,
        # Misc
        # ---------------------------------
        eval_model_index=0,
        # Hyper-parameters
        # ---------------------------------
        embedding_dim=FLAGS.embedding_dim,
        num_units=FLAGS.num_units,
        num_layers=FLAGS.num_layers,
        dropout_rate=FLAGS.dropout_rate,
        learning_rate=FLAGS.learning_rate,

        # Multi-Task
        # ---------------------------------
        training_strategy=FLAGS.training_strategy,
        embedding_type=FLAGS.model_type.split("-")[0],
        base_model_type=FLAGS.model_type.split("-")[1],
        multitask_model_type=FLAGS.model_type.split("-")[2],
        auto_model_type=(FLAGS.mixing_ratios if isAuto else None),
        mixing_ratios=(
            [int(r) for r in FLAGS.mixing_ratios.split("-")]
            if FLAGS.mixing_ratios is not None and not isAuto else None),

        # Multi-Task Hyperparams
        # ---------------------------------
        automr_update_rate=FLAGS.automr_update_rate,
        automr_reward_scale=FLAGS.automr_reward_scale,
        is_distill=FLAGS.is_distill,
        loss_coefficient_loc=FLAGS.loss_coefficient_loc,
        loss_coefficient_scale=FLAGS.loss_coefficient_scale,
        distill_temperature=FLAGS.distill_temperature
    )



def trainMTL(hparams):

    # Build Models and Data
    # ------------------------------------------
    # with misc_utils.suppress_stdout():
    train_MTL_model, val_MTL_model = model_utils.build_model(hparams)

    # building training monitor
    # ------------------------------------------
    # early stop on the **target** task
    eval_task = hparams.tasks[hparams.eval_model_index]
    manager = training_manager.TrainingManager(
        name=eval_task.name,
        logdir=hparams.manager_logdir,
        stopping_fn=eval_task.manager_stopping_fn(
            tolerance=EARLY_STOP_TOLERANCE),
        updating_fn=eval_task.manager_updating_fn(),
        load_when_possible=False)

    scores_dict = _train(
        hparams=hparams,
        manager=manager,
        train_MTL_model=train_MTL_model,
        val_MTL_model=val_MTL_model)

    # log the results for easier inspectation
    with open(hparams.train_logfile, "a") as f:
        for tag, score in scores_dict.items():
            f.write("%s: %.3f\t" % (tag, score))
        f.write("\n")

    print("FINISHED")


def _train(hparams, manager, train_MTL_model, val_MTL_model):
    # initialize *all* data generator
    # ------------------------------------------
    train_MTL_model.initialize_or_restore_session(
        ckpt_file=hparams.ckpt_file,
        var_filter_fn=lambda name: "Adam" not in name and "clone" not in name)
    train_MTL_model.initialize_data_iterator(model_idx=None)

    # TRAIN
    # ------------------------------------------
    pbar = trange(hparams.max_steps)
    for _ in pbar:
        try:
            _, message = train_MTL_model.train()
            pbar.set_description(message)
        except tf.errors.OutOfRangeError:
            raise ValueError("Task Finished An Epoch, this should not happen")

        # Evaluate the model
        # ------------------------------------------
        if train_MTL_model.global_step % hparams.steps_per_eval == 0:
            with misc_utils.suppress_stdout():
                ckpt = train_MTL_model.save_session()
                tf.logging.info("Running Evaluation")
                val_MTL_model.initialize_or_restore_session(
                    var_filter_fn=lambda name: "Adam" not in name)
                val_MTL_model.initialize_data_iterator(
                    [hparams.eval_model_index])
                scores_dict = val_MTL_model.evaluate(
                    model_idx=hparams.eval_model_index,
                    max_eval_batches=AUTOMR_MAX_EVAL_BATCHES)

                if multitask_models.is_AutoMR(train_MTL_model):
                    train_MTL_model.update_TaskSelector(scores_dict["MAIN"])

                # Log the best ckpt, which will be saved in a
                # different directory. Note that when manager.should_update
                # returns False, the manager.update will not do anything anyway
                if manager.should_update({"Scores": scores_dict["MAIN"]}):
                    ckpt = train_MTL_model.save_best_session()

                manager.update(value={"Scores": scores_dict["MAIN"]},
                               ckpt=ckpt, verbose=True)
                manager.save()

        if manager.should_stop:
            print("Manager has given the order to stop")
            pbar.close()
            break

    return manager.best_value


def infer(hparams):

    # Build Models and Data
    # ------------------------------------------
    _, infer_model = model_utils.build_model(hparams)

    manager = training_manager.TrainingManager(
        name=hparams.task_names[MAIN_MODEL_INDEX],
        logdir=hparams.manager_logdir)

    if hparams.ckpt_file is not None:
        ckpt_file = hparams.ckpt_file
        print("Using Specified CKPT from %s" % ckpt_file)

    else:
        ckpt_file = manager.best_checkpoint
        print("Using Manager CKPT from %s" % ckpt_file)

    if ckpt_file is None:
        raise ValueError("`ckpt_file` is None")

    tf.logging.info("Running Evaluation")
    infer_model.initialize_or_restore_session(
        ckpt_file=ckpt_file,
        var_filter_fn=lambda name: "Adam" not in name)
    infer_model.initialize_data_iterator()
    infer_model.inference(model_idx=MAIN_MODEL_INDEX)


def main(unused_argv):
    hparams = get_hparams()

    # set seed
    np.random.seed(hparams.numpy_seed)
    tf.set_random_seed(hparams.tensorflow_seed)

    if hparams.infer:
        infer(hparams)
    else:
        trainMTL(hparams)


if __name__ == "__main__":
    tf.app.run()
