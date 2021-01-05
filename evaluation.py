import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Model
from data.dataset import RelationExtractionDataset
from models.utils.checkpointing import load_checkpoint
from models.utils.scorer import score

class Evaluation(object):
	def __init__(self, hparams, model = None, split = "test", label2id=None):

		self.hparams = hparams
		self.model = model
		self._logger = logging.getLogger(__name__)
		self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
									 if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))
		self.split = split
		print("Evaluation Split :", self.split)
		do_valid, do_test = False, False
		if split == "dev":
			do_valid = True
		else:
			do_test = True
		self.label2id =label2id
		self.id2label = [label_key for label_key in self.label2id.keys()]
		self._build_dataloader(do_valid=do_valid, do_test=do_test)
		self._dataloader = self.valid_dataloader if split == 'dev' else self.test_dataloader

		if model is None:
			print(model)
			self._build_model()

	def _build_dataloader(self, do_valid=False, do_test=False):
		if self.label2id is None:
			self.train_dataset = RelationExtractionDataset(
				self.hparams,
				split="train",
			)
			self.label2id = self.train_dataset.label2id

		if do_valid:
			self.valid_dataset = RelationExtractionDataset(
				self.hparams,
				split="dev",
				label2id=self.label2id
			)
			self.valid_dataloader = DataLoader(
				self.valid_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
				collate_fn=self.valid_dataset.collate_fn
			)

		if do_test:
			self.test_dataset = RelationExtractionDataset(
				self.hparams,
				split="test",
				label2id=self.label2id
			)

			self.test_dataloader = DataLoader(
				self.test_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
				collate_fn=self.test_dataset.collate_fn
			)

	def _build_model(self):
		self.model = Model(self.hparams, len(self.label2id))
		self.model = self.model.to(self.device)
		# Use Multi-GPUs
		if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
			self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

	def run_evaluate(self, evaluation_path):
		self._logger.info("Evaluation")
		model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)

		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(model_state_dict)
		else:
			self.model.load_state_dict(model_state_dict)

		self.model.eval()

		ground_truth_str, predictions_str, ground_truth, predictions = [], [], [], []
		with torch.no_grad():
			for batch_idx, batch in enumerate(tqdm(self._dataloader)):
				buffer_batch = batch.copy()
				for key in batch:
					buffer_batch[key] = batch[key].to(self.device)

				logits = self.model(buffer_batch)
				pred = torch.argmax(logits, dim=1) # bs, num_classes

				ground_truth_str.extend([str(gt)]for gt in buffer_batch["relation"].to("cpu").tolist())
				predictions_str.extend([str(p)] for p in pred.to("cpu").tolist())

				ground_truth.extend(buffer_batch["relation"].to("cpu").tolist())
				predictions.extend(pred.to("cpu").tolist())

		ground_truth = [self.id2label[p] for p in ground_truth]
		predictions = [self.id2label[p] for p in predictions]
		# print("predictions : ", predictions)
		# print("ground_truth : ", ground_truth)

		p, r, f1 = score(ground_truth, predictions, verbose=True)
		self._logger.info("precision:{:4f}, recall:{:4f}, f1:{:4f}".format(p,r,f1))

