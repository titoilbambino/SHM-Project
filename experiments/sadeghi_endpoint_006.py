"""Evaluate Sadeghi Eshkevari's CNN and attention architecture on the 005 endpoint task."""

from bridge_shm.endpoint_experiment import run_from_cli


def main():
    from bridge_shm.sadeghi import build_sadeghi

    run_from_cli("006", __doc__, build_model=build_sadeghi)


if __name__ == "__main__":
    main()
