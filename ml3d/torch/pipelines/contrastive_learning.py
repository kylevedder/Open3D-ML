import torch
import logging
from tqdm import tqdm
import numpy as np
import re

from datetime import datetime
import matplotlib.pyplot as plt

from os.path import exists, join
from torch.utils.data import DataLoader
from pathlib import Path

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader, ConcatBatcher
from torch.utils.tensorboard import SummaryWriter
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, LogRecord, get_runid, code2md
from ...datasets.utils import BEVBox3D

from ...metrics.mAP import mAP

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ContrastiveLearning(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self,
                 model,
                 dataset=None,
                 name='ObjectDetection',
                 main_log_dir='./logs/',
                 device='cuda',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)

    def run_inference(self, data):
        pass

    def run_test(self):
        pass

    def run_valid(self, epoch=0):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        valid_dataset = dataset.get_split('validation')
        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      shuffle=True,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_valid', None))
        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        log.info("Started validation")
        slots_save_dir = cfg.slots_save_dir
        Path(f"{slots_save_dir}/").mkdir(parents=True, exist_ok=True)

        self.valid_losses = {}
        with torch.no_grad():
            for valid_idx, data in enumerate(
                    tqdm(valid_loader, desc='validation')):
                data.to(device)
                results = model(data)

                for batch_idx, slots in enumerate(results[0]):
                    slots = slots.cpu().numpy()
                    num_slots = slots.shape[0]
                    for i in range(num_slots):
                        plt.subplot(num_slots, 1, i + 1)
                        plt.imshow(slots[i], cmap='binary')
                        plt.colorbar()
                        plt.title(f"Slot {i}")
                    plt.savefig(
                        f"{slots_save_dir}/epoch_{epoch}_slots_{valid_idx}_{batch_idx}.png")
                    plt.clf()

                loss = model.loss(results, data)
                for l, v in loss.items():
                    if not l in self.valid_losses:
                        self.valid_losses[l] = []
                    self.valid_losses[l].append(v.cpu().numpy())

    def run_train(self):
        """Run training with train data split."""
        model = self.model
        device = self.device
        dataset = self.dataset

        cfg = self.cfg

        log.info("DEVICE : {}".format(device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        train_dataset = dataset.get_split('training')
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))
        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        self.optimizer, self.scheduler = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.tensorboard_dir))

        log.info("Started training")
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()

            self.losses = {}

            process_bar = tqdm(train_loader, desc='training')
            for data in process_bar:
                data.to(device)
                results = model(data)
                loss = model.loss(results, data)
                loss_sum = sum(loss.values())

                self.optimizer.zero_grad()
                loss_sum.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    model.cfg.grad_clip_norm)
                self.optimizer.step()
                desc = "training - "
                for l, v in loss.items():
                    if not l in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().detach().numpy())
                    desc += " %s: %.03f" % (l, v.cpu().detach().numpy())
                desc += " > loss: %.03f" % loss_sum.cpu().detach().numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

            if self.scheduler is not None:
                self.scheduler.step()

            # --------------------- validation
            if (epoch % cfg.get("validation_freq", 1)) == 0:
                self.run_valid(epoch)

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

    def save_logs(self, writer, epoch):
        for key, val in self.losses.items():
            writer.add_scalar("train/" + key, np.mean(val), epoch)

        if (epoch % self.cfg.get("validation_freq", 1)) == 0:
            for key, val in self.valid_losses.items():
                writer.add_scalar("valid/" + key, np.mean(val), epoch)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        epoch = 0
        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
                epoch = int(re.findall(r'\d+', ckpt_path)[-1]) + 1
            else:
                log.info('Initializing from scratch.')
                return epoch

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info(f'Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info(f'Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
            #scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary."""
        writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
        writer.add_text("Description/Command line", self.cfg_tb['cmd_line'], 0)
        writer.add_text('Configuration/Dataset',
                        code2md(self.cfg_tb['dataset'], language='json'), 0)
        writer.add_text('Configuration/Model',
                        code2md(self.cfg_tb['model'], language='json'), 0)
        writer.add_text('Configuration/Pipeline',
                        code2md(self.cfg_tb['pipeline'], language='json'), 0)


PIPELINE._register_module(ContrastiveLearning, "torch")
