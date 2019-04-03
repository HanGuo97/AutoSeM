from multitask.hard_sharing_model import MultitaskHardSharingModel
from multitask.multitask_autoMR_model import MTLAutoMRModel

def is_AutoMR(obj):
	return isinstance(obj, MTLAutoMRModel)
