"""Train the 005 temporal CNN on high-pass-filtered endpoint strain."""

from bridge_shm.endpoint_experiment import run_from_cli


def main():
    run_from_cli("007", __doc__)


if __name__ == "__main__":
    main()
