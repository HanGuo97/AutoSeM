from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from namedlist import namedlist
from collections import namedtuple
from utils import misc_utils
from bandits.base import BaseBandit


def _safe_standardization(reward, hists, ahists):
    if np.std(hists) < 1e-3:
        # here np.std() will lead to 0.0
        return 0.0

    return (reward - np.mean(hists)) / np.std(hists)


def _prediction_gain(reward, hists, ahists):
    if hists[-1] == 0.0:
        return 0.0

    return reward - hists[-1]


def _binary_prediction_gain(reward, hists, ahists):
    if hists[-1] == 0.0:
        return 0.0

    return float(reward > hists[-1])


def _binary_prediction_gain_v2(reward, hists, ahists):
    if hists[-1] == 0.0:
        return 0.0

    return float(reward >= hists[-1])


def _binary_prediction_gain_v3(reward, hists, ahists):
    # From Bandit Documentation
    # at the start, the update_histories is empty
    # to avoid nan, we will force set this to 0
    # if hists[-1] == 0.0:
    #     return 0.0

    return float(reward >= hists[-1])


class MTLAutoMRModel(object):
    """
    Multitask model with automatic task selection

    Build a TaskSelector object that keeps track of
    previous val-loss when running on task S, and
    choose which task to run by sampling from:

        S_t ~ P(S_t | history of validation loss)

    where P is modeled as a boltzmann distribution

        P(S | history) = softmax(history)

    and S is kept constant until new validation loss
    is available, that is, every 10 or so steps
    
    Q Score should be negative loss thus lower is better
    high initial Q for being "optimistic under uncertainty"
    """

    def __init__(self,
                 model,
                 initial_weight,
                 update_rate=0.3,
                 reward_scale=1.0,
                 temperature_anneal_rate=None):
        self._model = model
        # save new arguments
        self._initial_weight = initial_weight
        self._update_rate = update_rate
        self._reward_scale = reward_scale
        self._temperature_anneal_rate = temperature_anneal_rate
    
    def build(self):
        # build the MTL base model
        self._model.build()
        # build the Selector
        self._TaskSelector = BernoulliBanditTS(
            num_actions=self.num_models,
            reward_shaping_fn=lambda reward, hists, ahists: (
                _binary_prediction_gain_v3(reward, hists, ahists)),
            decay_rate=self._update_rate)

        # initial task will be main task
        self._selector_task_index = 0

    def update_TaskSelector(self, reward):
        
        self._TaskSelector.update(
            reward=reward,
            chosen_arm=self._selector_task_index)

        # sample a new task to run
        self._selector_task_index, _ = (
            self._TaskSelector.sample(step=self._model.global_step))

    def _AutoMR_task_selector(self, step):
        # override parent method
        # step argument is kept for compatability
        return self._selector_task_index

    def train(self):
        return self._model.train(
            model_idx=self._AutoMR_task_selector(self.global_step))

    def evaluate(self, *args, **kargs):
        # override parent method
        scores_dict = self._model.evaluate(*args, **kargs)
        
        if not isinstance(scores_dict, dict):
            raise TypeError("`scores_dict` is supposed to be dict")

        if "MAIN" not in scores_dict:
            raise ValueError("`MAIN` not in `scores_dict`")

        return scores_dict

    def inference(self, *args, **kargs):
        return self._model.inference(*args, **kargs)

    # Save and Load the Selector
    # ----------------------------------------------
    @property
    def selector_dir(self):
        return os.path.join(self._model._logdir, "mab_selector")

    def save_selector(self):
        # additionally save the selector
        self._TaskSelector.save(self.selector_dir)

    def load_selector(self):
        try:
            # additionally restore the selector
            self._TaskSelector.load(self.selector_dir)

        except ValueError:
            # the files haven't been created, skipping
            pass
    
    def save_session(self):
        self.save_selector()
        return self._model.save_session()

    def initialize_or_restore_session(self, *args, **kargs):
        self.load_selector()
        return self._model.initialize_or_restore_session(*args, **kargs)


    # Copy the interface (core class)
    # ----------------------------------------------
    @property
    def global_step(self):
        return self._model.global_step

    def write_summary(self, *args, **kargs):
        return self._model.write_summary(*args, **kargs)

    def save_best_session(self):
        return self._model.save_best_session()

    # Copy the interface (base class)
    # ----------------------------------------------
    def initialize_data_iterator(self, *args, **kargs):
        return self._model.initialize_data_iterator(*args, **kargs)

    @property
    def num_models(self):
        return self._model.num_models

    @property
    def num_tasks(self):
        return self._model.num_tasks

    @property
    def not_multitask(self):
        return self._model.not_multitask

    @property
    def total_steps(self):
        return self._model.total_steps

    @property
    def main_task_step(self):
        return self._model.main_task_step




Parameter = namedlist(
    "Parameter",
    ("Alpha", "Beta"))
SampleHistory = namedtuple(
    "SampleHistory",
    ("ChosenArm", "SampledValues"))
UpdateHistory = namedtuple(
    "UpdateHistory",
    ("Reward", "ShapedReward", "ChosenArm", "Parameters"))


def random_argmax(vector):
    """Helper function to select argmax at random... not just first one."""
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index


class BernoulliBanditTS(BaseBandit):

    def __init__(self,
                 num_actions,
                 reward_shaping_fn,
                 prior_alpha=1,
                 prior_beta=1,
                 decay_rate=0.0):
        """
        Args:
            prior_alpha:
                Prior parameters for Beta Distribution
            prior_beta:
                Prior parameters for Beta Distribution
            decay_rate:
                How quickly uncertainty is injected. Set to
                non-zero values will effectly create a non-stationary TS bandit

        """
        super(BernoulliBanditTS, self).__init__()
        if not callable(reward_shaping_fn):
            raise TypeError("`reward_shaping_fn` must be callable")

        self._num_actions = num_actions
        self._reward_shaping_fn = reward_shaping_fn
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._decay_rate = decay_rate
        self._parameters = [
            Parameter(Alpha=prior_alpha, Beta=prior_beta)
            for _ in range(num_actions)]

        self._sample_histories = []
        self._update_histories = []
        

    @property
    def alphas(self):
        return [p.Alpha for p in self._parameters]

    @property
    def betas(self):
        return [p.Beta for p in self._parameters]

    @property
    def arm_weights(self):
        return [p.Alpha / (p.Alpha + p.Beta) for p in self._parameters]

    def get_reward_histories(self, chosen_arm=None):
        histories = [h.Reward for h in self._update_histories]
        arm_histories = (
            [h.Reward for h in self._update_histories
             if h.ChosenArm == chosen_arm]
            if chosen_arm is not None else None)

        # at the start, the update_histories is empty
        # to avoid nan, we will force set this to 0
        if len(histories) == 0:
            histories = [0.0]
        if len(arm_histories) == 0:
            arm_histories = ([0.0] if chosen_arm is not None else None)

        return histories, arm_histories
    
    def sample(self, step=0):
        sampled_means = np.random.beta(self.alphas, self.betas)
        chosen_arm = random_argmax(sampled_means)

        self._sample_histories.append(
            SampleHistory(
                ChosenArm=chosen_arm,
                SampledValues=sampled_means))

        return chosen_arm, sampled_means

    def update(self, reward, chosen_arm):
        shaped_reward = self._reward_shaping_fn(
            reward, *self.get_reward_histories(chosen_arm))
        
        if shaped_reward not in [0.0, 1.0]:
            raise ValueError("`shaped_reward` should be a Bernoulli variable")

        # All values decay slightly, observation updated
        for arm in range(self._num_actions):
            # 􏰁(1 − \gamma) \alpha + \gamma \alpha_bar
            # (1 − \gamma) \beta + \gamma \beta_bar
            # where *_bar is a hyper-parameter which we chose
            # to be the prior (\alpha = \beta = 1), i.e. uniform
            self._parameters[arm].Alpha = (
                (1 - self._decay_rate) *
                self._parameters[arm].Alpha +
                self._decay_rate * self._prior_alpha)
            self._parameters[arm].Beta = (
                (1 - self._decay_rate) *
                self._parameters[arm].Beta +
                self._decay_rate * self._prior_beta)

        self._parameters[chosen_arm].Alpha += shaped_reward
        self._parameters[chosen_arm].Beta += 1 - shaped_reward

        parameter_snapshot = [
            {"Arm": i, "Alpha": p.Alpha, "Beta": p.Beta}
            for i, p in enumerate(self._parameters)]

        self._update_histories.append(
            UpdateHistory(
                Reward=reward,
                ShapedReward=shaped_reward,
                ChosenArm=chosen_arm,
                Parameters=parameter_snapshot))
        

    def save(self, file_dir):
        misc_utils.save_object(
            self._parameters, file_dir + "._parameters")
        misc_utils.save_object(
            self._sample_histories, file_dir + "._sample_histories")
        misc_utils.save_object(
            self._update_histories, file_dir + "._update_histories")

    def load(self, file_dir):
        try:
            self._parameters = misc_utils.load_object(
                file_dir + "._parameters")
            self._sample_histories = misc_utils.load_object(
                file_dir + "._sample_histories")
            self._update_histories = misc_utils.load_object(
                file_dir + "._update_histories")
        
        except FileNotFoundError:
            raise ValueError("%s File not exist ", file_dir)
