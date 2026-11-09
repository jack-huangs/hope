"""B/C 批跑兼容入口，默认选择 B 和 C。"""
from .run_batch import NoRetrieval, main as batch_main, run_batch


def run_bc(preparation, output, model, *, workers=4, attempts=2,
           max_calls=30000, max_tokens=300000000):
    return run_batch(preparation, output, model, modes=("b", "c"), workers=workers,
                     attempts=attempts, max_calls=max_calls, max_tokens=max_tokens)


def main(argv=None):
    return batch_main(argv, default_modes=("b", "c"), module_name="evidence_mdt.run_bc")


if __name__ == "__main__":
    raise SystemExit(main())
