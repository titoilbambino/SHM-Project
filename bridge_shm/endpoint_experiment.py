"""Recording-held-out training and evaluation for endpoint strain experiments."""

import argparse
from importlib.metadata import version
from math import ceil
from pathlib import Path
from statistics import median

from bridge_shm.data import (
    RecordingWindows, endpoint_indices, fit_channel_mean, highpass_strain, load_recordings,
    training_windows,
)
from bridge_shm.evaluation import mean_scores, regression_scores
from bridge_shm.runs import create_run_directory, read_config, write_json


ROOT = Path(__file__).resolve().parents[1]


def prepare(config, path):
    data, windows = config["data"], config["windows"]
    train, test = data["train"], data["test"]
    records = load_recordings(path, train + test, target=data["target"],
                              sample_rate_hz=data["sample_rate_hz"], feature_count=data["feature_count"],
                              target_unit=data["target_unit"])
    counts = {}
    for name, record in records.items():
        if "target_highpass" in data:
            record.strain = highpass_strain(record.strain, data["sample_rate_hz"], **data["target_highpass"])
        samples = len(record.strain)
        count = {"samples": samples, "evaluation_endpoints": len(endpoint_indices(
            samples, windows["length"], windows["evaluation_stride"]
        ))}
        if name in train:
            count["training_endpoints"] = len(endpoint_indices(
                samples, windows["length"], windows["train_stride"]
            ))
        counts[name] = count
    return records, counts


def run(config, records, directory, build_model=None):
    from bridge_shm.training import fit_tcn, predict

    data, windows, training = config["data"], config["windows"], config["training"]
    train = data["train"]
    device = training["device"]
    length, fit_stride = windows["length"], windows["train_stride"]
    evaluation_stride = windows["evaluation_stride"]
    fit_options = dict(n_features=data["feature_count"], architecture=config["model"],
                       training=training, device=device)
    if build_model is not None:
        fit_options["build_model"] = build_model
    folds = []
    for name in train:
        fit_names = [other for other in train if other != name]
        mean = fit_channel_mean(records, fit_names)
        fit = training_windows(records, fit_names, mean, length, fit_stride)
        val = RecordingWindows(records[name], mean, length, evaluation_stride)
        print(f"Validation recording: {name}", flush=True)
        model, best_update, history = fit_tcn(fit, validation=val, epochs=training["max_epochs"], **fit_options)
        scores = regression_scores(val.targets, predict(model, val, device))
        updates = ceil(len(fit) / training["batch_size"])
        folds.append(dict(recording=name, windows=len(val), best_epoch=best_update / updates,
                          updates_at_best=best_update, updates_run=history["validation_checks"][-1]["update"],
                          history=history, **scores))
        write_json(directory / "validation.json", folds)
        del model, fit, val

    final_epochs = round(median(row["best_epoch"] for row in folds))
    mean = fit_channel_mean(records, train)
    fit = training_windows(records, train, mean, length, fit_stride)
    print(f"Final fit: {final_epochs} epochs", flush=True)
    model, _, history = fit_tcn(fit, epochs=final_epochs, **fit_options)
    test = []
    for name in data["test"]:
        ds = RecordingWindows(records[name], mean, length, evaluation_stride)
        scores = regression_scores(ds.targets, predict(model, ds, device))
        test.append(dict(recording=name, windows=len(ds), **scores))
    result = dict(validation_mean=mean_scores(folds), test=test, test_mean=mean_scores(test),
                  epochs=final_epochs, updates=final_epochs * ceil(len(fit) / training["batch_size"]),
                  history=history)
    write_json(directory / "scores.json", result)
    print(f"Mean test RMSE: {result['test_mean']['rmse']:.4f} {data['target_unit']}, "
          f"mean test R²: {result['test_mean']['r2']:.4f}", flush=True)


def run_from_cli(number, description, build_model=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{number}.yaml")
    parser.add_argument("--data", type=Path, help="Override the HDF5 path from the config")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / number,
                        help="Parent directory for this experiment's runs")
    parser.add_argument("--check-data", action="store_true", help="Check recordings and counts, then exit")
    args = parser.parse_args()
    config = read_config(args.config)
    path = args.data or ROOT / config["data"]["path"]
    if not path.is_file():
        parser.error(f"Dataset missing at {path}. Use --data PATH to select the HDF5 file.")
    records, counts = prepare(config, path)
    for name, count in counts.items():
        print(name, count)
    print("Full-fit training endpoints:", sum(c.get("training_endpoints", 0) for c in counts.values()))
    print("Test endpoints:", sum(counts[n]["evaluation_endpoints"] for n in config["data"]["test"]))
    if args.check_data:
        return

    from bridge_shm.training import require_device
    require_device(config["training"]["device"])
    directory = create_run_directory(args.output)
    print(f"Run directory: {directory}", flush=True)
    versions = {name: version(name) for name in ("keras", "torch", "numpy", "h5py")}
    if "target_highpass" in config["data"]:
        versions["scipy"] = version("scipy")
    write_json(directory / "setup.json",
               dict(config=config, versions=versions, data_path=str(path.resolve()), counts=counts))
    run(config, records, directory, build_model=build_model)
