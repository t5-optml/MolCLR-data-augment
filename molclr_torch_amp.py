import os
import shutil
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import GradScaler, autocast

from utils.nt_xent import NTXentLoss


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml',
                    os.path.join(model_checkpoints_folder, 'config.yaml'))


class MolCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(
            self.device, config['batch_size'], **config['loss'])
    
        # Initialize GradScaler if fp16_precision is enabled
        if self.config["fp16_precision"]:
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def _get_device(self):
        if torch.cuda.is_available() and self.config["gpu"] != "cpu":
            device = self.config["gpu"]
            torch.cuda.set_device(device)
        else:
            device = "cpu"
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]
        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config["model_type"] == "gin":
            from models.ginet_molclr import GINet

            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config["model_type"] == "gcn":
            from models.gcn_molclr import GCN

            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            raise ValueError("Undefined GNN model.")
        print(model)

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'],
            eta_min=0, last_epoch=-1
        )

        model_checkpoints_folder = os.path.join(
            self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config["epochs"]):
            for bn, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # Use autocast for mixed precision if enabled
                with autocast("cuda", enabled=self.config["fp16_precision"]):
                    loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar(
                        'train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[
                                           0], global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                if self.config["fp16_precision"]:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config["eval_every_n_epochs"] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, "(validation)")
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoints_folder, "model.pth"),
                    )

                self.writer.add_scalar(
                    "validation_loss", valid_loss, global_step=valid_n_iter
                )
                valid_n_iter += 1

            if (epoch_counter + 1) % self.config["save_every_n_epochs"] == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        model_checkpoints_folder,
                        "model_{}.pth".format(str(epoch_counter)),
                    ),
                )

            # warmup for the first few epochs
            if epoch_counter >= self.config["warm_up"]:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(
                "./ckpt", self.config["load_model"], "checkpoints"
            )
            state_dict = torch.load(os.path.join(checkpoints_folder, "model.pth"))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            counter = 0
            for xis, xjs in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                with autocast("cuda", enabled=self.config["fp16_precision"]):
                    loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter

        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if config["aug"] == "node":
        from dataset.dataset import MoleculeDatasetWrapper
    elif config["aug"] == "subgraph":
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    elif config['aug'] == 'func':
        from dataset.dataset_func import MoleculeDatasetWrapper
    elif config['aug'] == 'func_excl':
        from dataset.dataset_func_excl import MoleculeDatasetWrapper
    elif config['aug'] == 'func_mix':
        from dataset.dataset_func_mix import MoleculeDatasetWrapper
    elif config['aug'] == 'func_repl':
        from dataset.dataset_func_repl import MoleculeDatasetWrapper
    elif config["aug"] == "mix":
        from dataset.dataset_mix import MoleculeDatasetWrapper
    elif config["aug"] == "mix_new":
        from dataset.dataset_mix_new import MoleculeDatasetWrapper
    elif config['aug'] == 'no_aug':
        from dataset.dataset_no_aug import MoleculeDatasetWrapper
    elif config['aug'] == 'node_mask':
        from dataset.node_masking import MoleculeDatasetWrapper
    elif config['aug'] == 'edge_mask':
        from dataset.edge_masking import MoleculeDatasetWrapper
    elif config['aug'] == 'node_edge_mask':
        from dataset.node_edge_masking import MoleculeDatasetWrapper
    elif config["aug"] == "pollution":
        from dataset.dataset_pollution import MoleculeDatasetWrapper
    elif config["aug"] == "pollution_v2":
        from dataset.dataset_pollution_v2 import MoleculeDatasetWrapper
    else:
        raise ValueError("Not defined molecule augmentation!")

    if config["fp16_precision"]:
        print("Mixed precision training enabled.")
    else:
        print("Mixed precision training disabled.")

    dataset = MoleculeDatasetWrapper(config["batch_size"], **config["dataset"])
    molclr = MolCLR(dataset, config)

    import time

    start = time.time()
    molclr.train()
    print("Training time:", time.time() - start)


if __name__ == "__main__":
    main()
