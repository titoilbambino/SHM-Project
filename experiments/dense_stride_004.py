"""Train 004 using every eligible strain endpoint in the fit recordings."""

from bridge_shm.endpoint_experiment import prepare, run, run_from_cli


def main():
    run_from_cli("004", __doc__)


if __name__ == "__main__":
    main()
