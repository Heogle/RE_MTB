import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import logging
import sys

from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from evaluation import Evaluation

from data.dataset import RelationExtractionDataset

from models.utils.checkpointing import CheckpointManager, load_checkpoint

from models import Model

class RelationExtraction(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)

  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    self.train_dataset = RelationExtractionDataset(self.hparams, split="train")
    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.train_batch_size,
      num_workers=self.hparams.cpu_workers,
      shuffle=True,
      collate_fn=self.train_dataset.collate_fn,
      drop_last=True
    )

    print("""
       # -------------------------------------------------------------------------
       #   DATALOADER FINISHED
       # -------------------------------------------------------------------------
       """)
    print("=====================  self.hparams.learning_rate : ", self.hparams.learning_rate)

  def _build_model(self):
    # =============================================================================
    #   MODEL : Standard, Mention Pooling, Entity Marker
    # =============================================================================
    print('\t* Building model...')
    self.model = Model(self.hparams, len(self.train_dataset.label2id))
    self.model = self.model.to(self.device)

    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

    # =============================================================================
    #   CRITERION
    # =============================================================================
    #relation_counter = self.train_dataset.relation_counter
    #class_weights = [sum(relation_counter.values()) / relation_counter[key] for key in relation_counter.keys()]
    #self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))

    self.criterion = nn.CrossEntropyLoss()

    self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                T_max=len(self.train_dataset)//self.hparams.virtual_batch_size)

    self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

  def _setup_training(self):
    if self.hparams.save_dirpath == 'checkpoints/':
      self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
    self.summary_writer = SummaryWriter(self.save_dirpath)
    self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)

    # If loading from checkpoint, adjust start epoch and load parameters.
    if self.hparams.load_pthpath == "":
      self.start_epoch = 1
    else:
      # "path/to/checkpoint_xx.pth" -> xx
      self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
      # self.start_epoch += 1
      self.start_epoch = 5
      model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict)
      else:
        self.model.load_state_dict(model_state_dict)
      self.optimizer.load_state_dict(optimizer_state_dict)
      self.previous_model_path = self.hparams.load_pthpath
      print("Loaded model from {}".format(self.hparams.load_pthpath))

    # self.summary_writer.flush()
    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def train(self):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self._build_dataloader()
    self._build_model()
    self._setup_training()

    # Evaluation Setup
    evaluation = Evaluation(self.hparams, model=self.model, split="dev", label2id=self.train_dataset.label2id)

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)

    train_begin = datetime.utcnow()  # New
    global_iteration_step = 0
    accumulate_loss = 0
    accu_count = 0
    for epoch in range(self.start_epoch, self.hparams.num_epochs):
      self.model.train()

      tqdm_batch_iterator = tqdm(self.train_dataloader)
      accumulate_batch = 0

      for batch_idx, batch in enumerate(tqdm_batch_iterator):
        buffer_batch = batch.copy()
        for key in batch:
          buffer_batch[key] = buffer_batch[key].to(self.device)

        logits = self.model(buffer_batch)
        loss = self.criterion(logits, buffer_batch["relation"])
        loss.backward()
        accumulate_loss += loss.item()
        accu_count += 1

        # TODO: virtual batch implementation
        accumulate_batch += buffer_batch["relation"].shape[0]
        if self.hparams.virtual_batch_size == accumulate_batch \
            or batch_idx == (len(self.train_dataset) // self.hparams.train_batch_size): # last batch

          self.optimizer.step()

          nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
          self.optimizer.zero_grad()
          accumulate_batch = 0

          global_iteration_step += 1

          description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
            datetime.utcnow() - train_begin,
            epoch,
            global_iteration_step, (accumulate_loss / accu_count),
            self.optimizer.param_groups[0]['lr'])
          tqdm_batch_iterator.set_description(description)
          # accumulate_loss_wirte = accumulate_loss/accu_count
          # self.summary_writer.add_scalar('Loss/train', accumulate_loss_wirte, global_step=global_iteration_step)
          # self.summary_writer.add_scalars(global_step=global_iteration_step)

        # # tensorboard
        # if global_iteration_step % self.hparams.tensorboard_step == 10:
        #   description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
        #     datetime.utcnow() - train_begin,
        #     epoch,
        #     global_iteration_step, (accumulate_loss / global_iteration_step),
        #     self.optimizer.param_groups[0]['lr'],
        #   )
        #   self._logger.info(description)

      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      self.checkpoint_manager.step(epoch)
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
      self._logger.info(self.previous_model_path)

      torch.cuda.empty_cache()
      self._logger.info("Evaluation after %d epoch" % epoch)
      evaluation.run_evaluate(self.previous_model_path)
      torch.cuda.empty_cache()
      self.scheduler.step()