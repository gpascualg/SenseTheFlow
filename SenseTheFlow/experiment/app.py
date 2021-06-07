import sys
from absl.app import _run_init, parse_flags_with_usage, _init_callbacks


def run(main=None, argv=None):
    args = _run_init(
        sys.argv if argv is None else argv,
        parse_flags_with_usage,
    )
    while _init_callbacks:
        callback = _init_callbacks.popleft()
        callback()

    return main(args)


def execute_battery(parameters_list, callback):
    def _as_gen():
        for i, argument_set in enumerate(parameters_list):
            yield run(lambda args: callback(i, args), argv=[sys.argv[0]] + [f'--{k}={v}' for k, v in argument_set.items()])

    return list(_as_gen())
