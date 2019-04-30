import argparse


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, prefix_chars='-', *args, **kwargs):
        self.registry = {}
        self.prefix_chars = prefix_chars
        self.default_prefix = '-' if '-' in prefix_chars else prefix_chars[0]

        super().__init__(*args, **kwargs, prefix_chars=prefix_chars, add_help=False)

    def add_condition_group(self, condition_name, help=None):
        self.registry[condition_name] = {'help': help, 'conditions': {}}

    def add_condition(self, condition_name, condition, arg_add_fn, group_help=None):
        if condition_name not in self.registry:
            self.registry[condition_name] = {'help': group_help, 'conditions': {}}

        self.registry[condition_name]['conditions'][condition] = arg_add_fn

    def parse_args(self, args=None, namespace=None):
        preparsed, _ = self.parse_known_args(args, namespace)

        for cond_group, conds in self.registry.items():
            arg_group = getattr(preparsed, cond_group, None)

            if arg_group is not None:
                group = self.add_argument_group(
                    cond_group,
                    description=conds['help'],
                    argument_default=argparse.SUPPRESS,
                )
                conds['conditions'][arg_group].add_args(group)

        self.add_argument(
            self.default_prefix + 'h',
            self.default_prefix * 2 + 'help',
            action='help',
            default=argparse.SUPPRESS,
            help='show this help message and exit',
        )

        args = super().parse_args(args, namespace)

        return args
