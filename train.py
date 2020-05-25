import copy
import logging
import random
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from beautifultable import BeautifulTable
from sklearn.model_selection import StratifiedKFold, train_test_split

import diagnostics
from datasets import ChestXRayPneumoniaDataset  # NIHCX38Dataset
from datasets import COVIDChestXRayDataset
from models import Resnet34
from trainer import Trainer

# Run diagnostics
diagnostics.run()


def main(
    experiment_dir, baseline_epochs=20, finetune_epochs=15, seed=None, batch_size=64, image_size=256, n_splits=None
):
    """
    Main training loop

    Parameters
    ----------
    seed : int or None, optional
    batch_size : int, optional
    image_size : int, optional
    n_splits : int or None, optional
        If None, the model will be trained on all the available data. Default is None.
    """
    # Create experiment dir and checkpoints dir
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Set up root logger
    logger_path = experiment_dir / "train.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(name)s:%(lineno)d %(levelname)s :: %(message)s")
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Get train logger
    logger = logging.getLogger("defeatcovid19.train")

    if seed is not None:
        # Fix seed to improve reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Pretrain with Chest XRay Pneumonia dataset (>5k images)
    pneumonia_classifier = Resnet34()

    dataset = ChestXRayPneumoniaDataset(Path("/data/chest_xray_pneumonia"), image_size)
    dataset.build()

    # dataset = NIHCX38Dataset(Path('/data/nih-cx38'), size, balance=True)
    # dataset.build()

    train_idx, validation_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)
    trainer = Trainer(
        "baseline_classifier", pneumonia_classifier, dataset, batch_size, train_idx, validation_idx, checkpoints_dir
    )
    trainer.run(max_epochs=baseline_epochs)

    # Fine tune with COVID-19 Chest XRay dataset (~120 images)
    dataset = COVIDChestXRayDataset(Path("/data/covid-chestxray-dataset"), image_size)
    dataset.build()

    if n_splits is not None:
        logger.info(f"Executing a {n_splits}-fold cross validation")
        kfold_metrics = {
            "train": {"loss": [], "roc": [], "accuracy": []},
            "val": {"loss": [], "roc": [], "accuracy": []},
        }
        split = 1
        skf = StratifiedKFold(n_splits=n_splits)
        for train_idx, validation_idx in skf.split(dataset.df, dataset.labels):
            logger.info("===Split #{}===".format(split))
            # Start from the pneumonia classifier
            classifier = copy.deepcopy(pneumonia_classifier)
            # Checkpoints per split
            checkpoints_dir_split = checkpoints_dir / f"split_{split}"
            checkpoints_dir_split.mkdir(exist_ok=True)
            trainer = Trainer(
                "covid19_classifier", classifier, dataset, batch_size, train_idx, validation_idx, checkpoints_dir_split
            )
            trainer_metrics = trainer.run(max_epochs=finetune_epochs)

            # Record metrics for the current split
            for data_split_id, data_split_metrics in trainer_metrics.items():
                for metric_id, metric in data_split_metrics.items():
                    kfold_metrics[data_split_id][metric_id].append(metric)

            split += 1

        # Summarize metrics from all splits and compute mean and std
        table = BeautifulTable()
        table.column_headers = (
            ["Metric name"] + [f"Split {split_num+1}" for split_num in range(n_splits)] + ["Mean", "Std"]
        )
        for data_split_id, data_split_metrics in kfold_metrics.items():
            for metric_id, metric in data_split_metrics.items():
                metric_vals = kfold_metrics[data_split_id][metric_id]
                mean_metric = np.mean(metric_vals)
                std_metric = np.std(metric_vals)
                table_row = [f"{data_split_id} {metric_id}"] + metric_vals + [mean_metric, std_metric]
                table.append_row(table_row)
        logger.info(f"SUMMARY\n{table}")

    else:
        logger.info("Training with a fixed split")
        # Train / test split for covid data
        train_idx, validation_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels)
        # Start from the pneumonia classifier
        classifier = copy.deepcopy(pneumonia_classifier)
        trainer = Trainer(
            "covid19_classifier", classifier, dataset, batch_size, train_idx, validation_idx, checkpoints_dir
        )
        trainer_metrics = trainer.run(max_epochs=15)

        # Summarize metrics from training
        table = BeautifulTable()
        table.column_headers = ["Metric name", "Metric value"]
        for data_split_id, data_split_metrics in trainer_metrics.items():
            for metric_id, metric in data_split_metrics.items():
                table_row = [f"{data_split_id} {metric_id}", metric]
                table.append_row(table_row)
        logger.info(f"SUMMARY\n{table}")


if __name__ == "__main__":
    fire.Fire(main)
