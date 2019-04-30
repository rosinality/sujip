from .scheduler import CycleScheduler, LRFinder


class CycleSchedulerArgs:
    @classmethod
    def invoke(cls, args):
        return lambda optimizer: CycleScheduler(optimizer, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--cyc_mom',
            nargs=2,
            type=float,
            default=[0.95, 0.85],
            help='cycle momentum ranges',
        )
        parser.add_argument('--div', type=int, default=25, help='lr divider')
        parser.add_argument(
            '--warmup', type=float, default=0.3, help='proportion of warmup'
        )

    @staticmethod
    def get_args(args):
        return {
            'lr_max': args.lr,
            'n_iter': args.iter,
            'momentum': args.cyc_mom,
            'divider': args.div,
            'warmup_proportion': args.warmup,
        }


class LRFinderArgs:
    @classmethod
    def invoke(cls, args):
        return lambda optimizer: LRFinder(optimizer, **cls.get_args(args))

    @staticmethod
    def add_args(parser):
        parser.add_argument('--lr_min', type=float)
        parser.add_argument('--linear', action='store_true', default=False)

    @staticmethod
    def get_args(args):
        return {
            'lr_min': args.lr_min,
            'lr_max': args.lr,
            'step_size': args.iter,
            'linear': args.linear,
        }
