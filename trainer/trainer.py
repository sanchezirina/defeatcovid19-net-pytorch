import logging
import math
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from dataloaders import SubsetRandomDataLoader
from metrics import Accuracy


class Trainer:
    def __init__(self, name, classifier, dataset, batch_size, train_idx, validation_idx, checkpoints_dir=None):
        self.logger = logging.getLogger(__name__)

        self.name = name
        self.classifier = classifier
        self.batch_size = batch_size
        self.logger.info(
            f"Trainer '{self.name}' started with:\n "
            f"Classifier: {classifier.__class__.__name__}\n "
            f"Dataset: {dataset.__class__.__name__}\n "
            f"Batch size: {batch_size}"
        )

        self.optimizer = None
        self.scheduler = None

        self.train_dataset = dataset
        self.validation_dataset = dataset

        # train_idx, validation_idx = train_test_split(
        #     list(range(len(self.train_dataset))),
        #     test_size=0.2,
        #     stratify=self.train_dataset.labels
        # )

        self.train_loader = SubsetRandomDataLoader(dataset, train_idx, batch_size)
        self.validation_loader = SubsetRandomDataLoader(dataset, validation_idx, batch_size)

        self.logger.info("Train set: {}".format(len(train_idx)))
        self.logger.info("Validation set: {}".format(len(validation_idx)))

        self.it_per_epoch = math.ceil(len(train_idx) / self.batch_size)
        self.logger.info("Training with {} mini-batches per epoch".format(self.it_per_epoch))

        self.checkpoints_dir = Path(checkpoints_dir or "checkpoints/")

    def run(self, max_epochs=10, lr=0.01):
        self.classifier = self.classifier.cuda()
        model = self.classifier

        it = 0
        epoch = 0
        it_save = self.it_per_epoch * 5
        it_log = math.ceil(self.it_per_epoch / 5)
        it_smooth = self.it_per_epoch
        self.logger.info("Logging performance every {} iter, smoothing every: {} iter".format(it_log, it_smooth))

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 2 * self.it_per_epoch, gamma=0.9)

        criterion = nn.BCELoss()
        criterion = criterion.cuda()
        metrics = [Accuracy(), roc_auc_score]

        self.logger.info("{}'".format(self.optimizer))
        self.logger.info("{}'".format(self.scheduler))
        self.logger.info("{}'".format(criterion))
        self.logger.info("{}'".format(metrics))

        train_loss = 0
        train_roc = 0
        train_acc = 0

        self.logger.info("                    |         VALID        |        TRAIN         |         ")
        self.logger.info(" lr     iter  epoch | loss    roc    acc   | loss    roc    acc   |  time   ")
        self.logger.info("------------------------------------------------------------------------------")

        start = timer()
        while epoch < max_epochs:
            epoch_labels = []
            epoch_preds = []
            epoch_losses = []
            epoch_loss_weights = []

            for inputs, labels in self.train_loader:
                epoch = (it + 1) / self.it_per_epoch

                lr = self.scheduler.get_last_lr()[0]

                # checkpoint
                if it % it_save == 0 and it != 0:
                    self.save(model, self.optimizer, it, epoch)

                # training

                model.train()
                inputs = inputs.cuda().float()
                labels = labels.cuda().float()

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_labels.append(labels.cpu())
                epoch_preds.append(preds.cpu())
                epoch_losses.append(loss.data.cpu().numpy())
                epoch_loss_weights.append(len(preds))

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                if it % it_log == 0:
                    self.logger.info(
                        "{:5f} {:4d} {:5.1f} |                      |                      | {:6.2f}".format(
                            lr, it, epoch, timer() - start
                        )
                    )

                it += 1

            # loss and metrics
            with torch.no_grad():
                epoch_labels = torch.cat(epoch_labels)
                epoch_preds = torch.cat(epoch_preds)
                train_acc, train_roc = [i(epoch_labels.cpu(), epoch_preds.cpu()).item() for i in metrics]

            train_loss = np.average(epoch_losses, weights=epoch_loss_weights)

            # validation
            valid_loss, valid_m = self.do_valid(model, criterion, metrics)
            valid_acc, valid_roc = valid_m

            self.logger.info(
                "{:5f} {:4d} {:5.1f} | {:0.3f}* {:0.3f}  {:0.3f}  | {:0.3f}* {:0.3f}  {:0.3f}  | {:6.2f}".format(
                    lr, it, epoch, valid_loss, valid_roc, valid_acc, train_loss, train_roc, train_acc, timer() - start
                )
            )

            # Data loader end
        # Training end

        # Save checkpoints
        self.save(model, self.optimizer, it, epoch)

        # Return results
        return {
            "train": {"loss": train_loss, "roc": train_roc, "accuracy": train_acc},
            "val": {"loss": valid_loss, "roc": valid_roc, "accuracy": valid_acc},
        }

    def do_valid(self, model, criterion, metrics):
        model.eval()
        valid_num = 0
        valid_labels = []
        valid_preds = []
        losses = []
        loss_weights = []

        for inputs, labels in self.validation_loader:
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()

            with torch.no_grad():
                preds = model(inputs)
                loss = criterion(preds, labels)

            valid_num += len(inputs)
            valid_labels.append(labels.cpu())
            valid_preds.append(preds.cpu())
            losses.append(loss.data.cpu().numpy())
            loss_weights.append(len(inputs))

        assert valid_num == len(self.validation_loader.sampler)

        with torch.no_grad():
            valid_labels = torch.cat(valid_labels)
            valid_preds = torch.cat(valid_preds)
            m = [i(valid_labels.cpu(), valid_preds.cpu()).item() for i in metrics]

        loss = np.average(losses, weights=loss_weights)
        return loss, m

    def save(self, model, optimizer, iter, epoch):
        checkpoints_path = self.checkpoints_dir / f"{self.name}-{iter}_model.pth"
        optimizer_path = self.checkpoints_dir / f"{self.name}-{iter}_optimizer.pth"
        torch.save(model.state_dict(), str(checkpoints_path))
        torch.save(
            {"optimizer": optimizer.state_dict(), "iter": iter, "epoch": epoch}, str(optimizer_path),
        )
