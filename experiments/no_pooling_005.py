"""Train the 004 endpoint model without temporal max pooling."""

from bridge_shm.endpoint_experiment import run_from_cli


def main():
    run_from_cli("005", __doc__)


if __name__ == "__main__":
    main()
