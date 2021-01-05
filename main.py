import os
import sys
import argparse
import collections
import logging
from datetime import datetime
from train import RelationExtraction

from setproctitle import setproctitle
setproctitle("Heogle_RE")

from data.tacred_data_utils import InputExamples

from config.hparams import *

PARAMS_MAP = {
  "standard" : STANDARD_PARAMS,
  "mention_pooling" : MENTION_POOLING_PARAMS,
  "entity_markers" : ENTITY_MARKERS_PARAMS,
}

def init_logger(path:str):
  if not os.path.exists(path):
      os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger

def train_model(args):
    hparams = PARAMS_MAP[args.model]
    root_dir = hparams["root_dir"]
    hparams.update(root_dir=root_dir)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams["root_dir"], "%s/" % timestamp)
    logger = init_logger(root_dir)
    logger.info("Hyper-parameters: %s" % str(hparams))
    hparams["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

    model = RelationExtraction(hparams)
    model.train()

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Bert / Run Classifier (Tensorflow)")
  arg_parser.add_argument("--model", dest="model", type=str, default="entity_markers",
                          help="Model Name")
  arg_parser.add_argument("--data", dest="data", type=str, default="tacred",
                          help="dataset type")
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str, default="/mtb/checkpoints",
                          help="checkpoint path")
  args = arg_parser.parse_args()
  train_model(args)
  # CUDA_VISIBLE_DEVICES=3 python main.py --model entity_markers --data tacred