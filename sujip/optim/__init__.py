from .updater import Updater
from .lamb import LAMB
from .qhadam import QHAdam
from .scheduler import CycleAnnealScheduler, CycleScheduler

from .scheduler_args import CycleSchedulerArgs, LRFinderArgs
from .optim_args import AdamArgs


SCHEDULER_REGISTRY = {'cycle': CycleSchedulerArgs, 'lr_find': LRFinderArgs}


def _add_args(parser, name='sched', registry=SCHEDULER_REGISTRY):
    parser.add_condition_group(name)

    for cond, arg_cls in registry.items():
        parser.add_condition(name, cond, arg_cls)


def _get_object(args, name='sched', registry=SCHEDULER_REGISTRY):
    sched = getattr(args, name)

    if sched is not None:
        return registry[sched].invoke(args)

    else:
        return None


def add_scheduler_args(parser, name='sched', registry=SCHEDULER_REGISTRY):
    _add_args(parser, name, registry)


def get_scheduler(args, name='sched', registry=SCHEDULER_REGISTRY):
    return _get_object(args, name=name, registry=registry)


OPTIMIZER_REGISTRY = {'adam': AdamArgs}


def add_optimizer_args(parser, name='optim', registry=OPTIMIZER_REGISTRY):
    _add_args(parser, name, registry)


def get_optimizer(args, name='optim', registry=OPTIMIZER_REGISTRY):
    return _get_object(args, name=name, registry=registry)
