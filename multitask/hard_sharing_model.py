from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# import numpy as np
import tensorflow as tf
from multitask import multitask_base_model


class MultitaskHardSharingModel(
        multitask_base_model.MultitaskSingleAndDualStreamBaseModel):
    """Multitask Model with Hard Parameter Sharing"""
    def _check_compatability(self):
        pass

    def _build_models(self):

        logits_collections = []
        predictions_collections = []
        step_collections = {"GlobalStep": 0}
        loss_collections = []
        summary_collections = []
        update_variables = None

        for task_idx in range(self.num_models):
            scope_name = "Model_%s" % self._names[task_idx]
            model_name = "%s-%d" % (self._names[task_idx], task_idx)
            with tf.variable_scope(scope_name):
                (logits,
                 predictions,
                 cross_entropy) = self._build_single_model(
                    task_name=self._names[task_idx],
                    data=self._data[task_idx],
                    embedding_fn=self._embedding_fns[self._main_model_index],
                    encoder_fn_1=self._encoder_fns_1[self._main_model_index],
                    encoder_fn_2=self._encoder_fns_2[self._main_model_index],
                    logits_fn=self._logits_fns[task_idx])

                step_collections[model_name] = 0
                logits_collections.append(logits)
                predictions_collections.append(predictions)

                loss = cross_entropy
                summaries = [tf.summary.scalar(
                    name=self._names[task_idx] + "_XE_loss",
                    tensor=cross_entropy)]

                loss_collections.append(loss)
                summary_collections.append(summaries)


        return (logits_collections,
                predictions_collections,
                step_collections,
                loss_collections,
                summary_collections,
                update_variables)


    # def evaluate(self, model_idx,
    #              max_eval_batches=None,
    #              write_results=False,
    #              write_to_summary=True):
    #     """Override original evaluate with evaluate_auxiliary"""

    #     # Initialize all data generator, this will override
    #     # the initialization called before model.evaluate
    #     self.initialize_data_iterator()

    #     # write_summary needs global_step
    #     self._step_collections["GlobalStep"] = (
    #         self._sess.run(self._global_step_tensor))

    #     scores_dict = {}

    #     for task_idx in range(self.num_models):
    #         summary_name_prefix = "TASK/%s-%d/" % (
    #             self._names[task_idx], task_idx)

    #         counts, scores = self._evaluate(
    #             data=self._data[task_idx],
    #             predictions=self._predictions_collections[task_idx],
    #             evaluation_fn=self._evaluation_fns[task_idx],
    #             max_eval_batches=max_eval_batches,
    #             write_results=(write_results and
    #                            task_idx == self._main_model_index))

    #         scores_dict[summary_name_prefix] = scores
    #         if task_idx == self._main_model_index:
    #             scores_dict["MAIN"] = scores
            
    #         # write summary
    #         if write_to_summary:
    #             summ_name = summary_name_prefix + "ValScores"
    #             self.write_summary(summ_name, scores)

    #     multitask_scores = np.mean(list(scores_dict.values()))
    #     scores_dict["MultitaskScores"] = multitask_scores

    #     if write_to_summary:
    #         summ_name = "TASK/MultitaskScores"
    #         self.write_summary(summ_name, multitask_scores)

    #     return scores_dict
