from torch import optim

from .adamw import AdamW
from .qhadam import QHAdam
from .lamb import LAMB


class SGDArgs:
    @classmethod
    def invoke(cls, args):
        return lambda parameters: optim.SGD(parameters, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument('--momentum', type=float, default=0, help='SGD momentum')
        parser.add_argument(
            '--nesterov', type='store_true', default=False, help='Use nesterov momentum'
        )

    @staticmethod
    def get_args(args):
        return {
            'lr': args.lr,
            'momentum': args.momentum,
            'nesterov': args.nesterov,
            'weight_decay': args.l2,
        }


class AdamArgs:
    @classmethod
    def invoke(cls, args):
        return lambda parameters: optim.Adam(parameters, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--betas',
            nargs=2,
            type=float,
            default=[0.9, 0.999],
            help='Adam momentum (b1, b2)',
        )
        parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')

    @staticmethod
    def get_args(args):
        return {
            'lr': args.lr,
            'betas': args.betas,
            'eps': args.eps,
            'weight_decay': args.l2,
        }


class AdamWArgs:
    @classmethod
    def invoke(cls, args):
        return lambda parameters: AdamW(parameters, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--betas',
            nargs=2,
            type=float,
            default=[0.9, 0.999],
            help='AdamW momentum (b1, b2)',
        )
        parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon')

    @staticmethod
    def get_args(args):
        return {
            'lr': args.lr,
            'betas': args.betas,
            'eps': args.eps,
            'weight_decay': args.l2,
        }


class LAMBArgs:
    @classmethod
    def invoke(cls, args):
        return lambda parameters: optim.LAMB(parameters, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--betas',
            nargs=2,
            type=float,
            default=[0.9, 0.999],
            help='LAMB momentum (b1, b2)',
        )
        parser.add_argument('--eps', type=float, default=1e-8, help='LAMB epsilon')

    @staticmethod
    def get_args(args):
        return {
            'lr': args.lr,
            'betas': args.betas,
            'eps': args.eps,
            'weight_decay': args.l2,
        }


class QHAdamArgs:
    @classmethod
    def invoke(cls, args):
        return lambda parameters: QHAdam(parameters, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--betas',
            nargs=2,
            type=float,
            default=[0.9, 0.999],
            help='QHAdam momentum (b1, b2)',
        )
        parser.add_argument(
            '--nus',
            nargs=2,
            type=float,
            default=[1.0, 1.0],
            help='QHAdam nus (nu1, nu2)',
        )
        parser.add_argument('--eps', type=float, default=1e-8, help='QHAdam epsilon')

    @staticmethod
    def get_args(args):
        return {
            'lr': args.lr,
            'betas': args.betas,
            'nus': args.nus,
            'eps': args.eps,
            'weight_decay': args.l2,
        }
