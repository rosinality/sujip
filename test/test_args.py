from sujip import ArgumentParser
from sujip.optim import get_scheduler, add_scheduler_args

from torch import nn, optim


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--iter', type=int)

    return parser


def test_argparse_sanity():
    parser = get_parser()

    try:
        parser.parse_args('--sched cycle'.split())

        assert False

    except:
        pass


def test_scheduler_args():
    parser = get_parser()
    parser.add_argument('--sched')
    add_scheduler_args(parser)
    args = parser.parse_args('--lr 5e-4 --iter 1000 --sched cycle --warmup 0.1'.split())

    assert args.warmup == 0.1


def test_scheduler_invoke():
    parser = get_parser()
    parser.add_argument('--sched')
    add_scheduler_args(parser)
    args = parser.parse_args(
        '--lr 5e-4 --iter 1000 --sched lr_find --lr_min 0.1'.split()
    )
    sched = get_scheduler(args)(optim.SGD(nn.Linear(1, 1).parameters(), lr=1))

    assert sched.lr_min == 0.1
    assert sched.linear is False
