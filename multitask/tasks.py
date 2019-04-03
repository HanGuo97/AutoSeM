from sklearn import metrics
from constants import BASE_CACHED_DATA_DIR as BASE_DATA_DIR
from constants import (BATCH_SIZE,
                       STEPS_PER_EVAL,
                       EVAL_BATCH_MULTIPLIER)
_PROBLEMS = {}


def default_name(obj_class):
    """Convert a class name to the registry's default name for the class.
    Args:
        obj_class: the name of a class
    Returns:
        The registry's default name for the class.
    """
    return obj_class.__name__


def register_problem(name=None):
    """Register a Problem. name defaults to cls name snake-cased."""

    def decorator(p_cls, registration_name=None):
        """Registers & returns p_cls with registration_name or default name."""
        p_name = registration_name or default_name(p_cls)
        if p_name in _PROBLEMS:
            raise LookupError("Problem %s already registered." % p_name)

        _PROBLEMS[p_name] = p_cls
        p_cls.name = p_name
        return p_cls

    # Handle if decorator was used without parens
    if callable(name):
        p_cls = name
        return decorator(p_cls, registration_name=default_name(p_cls))

    return lambda p_cls: decorator(p_cls, name)


def list_problems():
    return sorted(list(_PROBLEMS))


def problem(name):
    """Retrieve a problem by name."""
    if name not in _PROBLEMS:
        all_problem_names = list_problems()
        error_lines = ["%s not in the set of supported problems:" % name
                       ] + all_problem_names
        error_msg = "\n  * ".join(error_lines)
        raise LookupError(error_msg)
    return _PROBLEMS[name]()


# =========================================================
# Implementations
# =========================================================

def _stop_by_tolerance(tolerance=5):
    def _stopping_fn(best, history):
        recent_history = history["Scores"][-tolerance:]
        
        # when the results plateau, i.e. not changing
        if (len(recent_history) == tolerance and
                all([best["Scores"] == h for h in recent_history])):
            return True

        return best["Scores"] not in recent_history
    # print("Early Stop with Tolerance %d" % tolerance)
    return _stopping_fn


def _stop_by_max_steps():
    def _stopping_fn(best, history):
        return False

    return _stopping_fn


def _stop_by_tolerance_with_warmup(tolerance=5, warmup_period=10):
    """Will not trigger stop signal before warmup_period ends"""
    def _stopping_fn(best, history):
        # warm up
        if len(history["Scores"]) < warmup_period:
            return False

        # normal stop rule
        return best["Scores"] not in history["Scores"][-tolerance:]

    # print("Early Stop with Tolerance %d" % tolerance)
    return _stopping_fn


def _greedy_update():
    def _updating_fn(best, history, value):
        return value["Scores"] > best["Scores"]
    return _updating_fn


def _greedy_update_with_warmup(warmup_period=10):
    """Always update before warmup_period ends"""
    def _updating_fn(best, history, value):
        # warm up
        if len(history["Scores"]) < warmup_period:
            return True

        # normal update rule
        # using >= fixes some issues when many results are same
        # and thus causing lost checkpoints
        return value["Scores"] >= best["Scores"]
        
    return _updating_fn



class Problem(object):

    def __init__(self, name,
                 batch_size=BATCH_SIZE,
                 max_steps=None,
                 steps_per_eval=STEPS_PER_EVAL):
        self._name = name
        self._batch_size = batch_size
        self._max_steps = max_steps
        self._steps_per_eval = steps_per_eval

    @property
    def name(self):
        return self._name

    def _generate_data(self, mode):
        return BASE_DATA_DIR + self._name + mode

    @property
    def train_data(self):
        return self._generate_data("train")

    @property
    def val_data(self):
        return self._generate_data("val")

    @property
    def test_data(self):
        return self._generate_data("test")

    @property
    def infer_data(self):
        return self._generate_data("test")
    

    @property
    def train_batch_size(self):
        return self._batch_size

    @property
    def evaluate_batch_size(self):
        return self._batch_size * EVAL_BATCH_MULTIPLIER

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def steps_per_eval(self):
        return self._steps_per_eval

    # =========================================================
    # Training Managers
    # =========================================================
    def manager_updating_fn(self, *args, **kwargs):
        return _greedy_update(*args, **kwargs)

    def manager_stopping_fn(self, *args, **kwargs):
        return _stop_by_tolerance(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


# =========================================================
# Specific Instances
# =========================================================
@register_problem
class SST(Problem):

    def __init__(self):
        super(SST, self).__init__(name="SST-2")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)
    
    def manager_stopping_fn(self, *args, **kwargs):
        return _stop_by_max_steps(1000000)


@register_problem
class CoLA(Problem):

    def __init__(self):
        super(CoLA, self).__init__(name="CoLA")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class MNLIMisMatched(Problem):

    def __init__(self):
        super(MNLIMisMatched, self).__init__(name="MNLI")

    @property
    def val_data(self):
        return BASE_DATA_DIR + self._name + "/val_mismatched"

    @property
    def test_data(self):
        return BASE_DATA_DIR + self._name + "/test_mismatched"
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class MNLIMatched(Problem):

    def __init__(self):
        super(MNLIMatched, self).__init__(name="MNLI")

    @property
    def val_data(self):
        return BASE_DATA_DIR + self._name + "/val_matched"

    @property
    def test_data(self):
        return BASE_DATA_DIR + self._name + "/test_matched"
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class QNLI(Problem):

    def __init__(self):
        super(QNLI, self).__init__(name="QNLI")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)

    def manager_stopping_fn(self, *args, **kwargs):
        return _stop_by_max_steps()


@register_problem
class RTE(Problem):

    def __init__(self):
        super(RTE, self).__init__(name="RTE")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class MRPC(Problem):

    def __init__(self):
        super(MRPC, self).__init__(name="MRPC")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class WNLI(Problem):

    def __init__(self):
        super(WNLI, self).__init__(name="WNLI")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)


@register_problem
class QQP(Problem):

    def __init__(self):
        super(QQP, self).__init__(name="QQP")
    
    def evaluate(self, pred, seq_1, seq_2, target):
        return metrics.accuracy_score(target, pred)
