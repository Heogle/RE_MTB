import os
import torch
import pickle
import sys

from torch.utils.data import Dataset
# from models.bert import tokenization_bert
# from models.albert import tokenization_albert
# from models.xlnet import tokenization_xlnet
from models.electra import tokenization_electra

from data.tacred_data_utils import *

class RelationExtractionDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, ground truth answer, answer options, dense annotations etc.
    """
    def __init__(
        self,
        hparams,
        split: str = "",
        label2id=None
    ):

        super().__init__()

        self.hparams = hparams
        self.split = split
        print("hparams.data_dir : ", hparams.data_dir)
        print("hparams.task_name, split : ", hparams.task_name, split)

        # read pkls -> Input Examples
        with open(hparams.data_dir % (hparams.task_name, split), "rb") as pkl_handle:
          print("pkl_handle : ", pkl_handle)
          self.input_examples = pickle.load(pkl_handle)

        # label2id
        if split == "train":
          self.relation_counter = dict()
          self.label2id = dict()
          for example in self.input_examples:
            try:
              self.relation_counter[example.relation] += 1
            except KeyError:
              self.relation_counter[example.relation] = 1
              self.label2id[example.relation] = len(self.label2id)
        else:
          self.label2id = label2id

        print("total %s examples" % split, len(self.input_examples))
        print("total %s relations" % split, self.label2id)

        # bert 수정
        # # bert_pretrained_dir = "/mnt/raid5/shared/bert/pytorch/%s/" % self.hparams.bert_pretrained
        # bert_pretrained_dir = "/home/heogle/%s/" % self.hparams.bert_pretrained
        # vocab_file_path = "%s-vocab.txt" % self.hparams.bert_pretrained
        # self._bert_tokenizer = tokenization_bert.BertTokenizer(
        #     vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
        #     do_lower_case=False
        # )

        # albert 수정
        # bert_pretrained_dir = "/home/heogle/%s/" % self.hparams.bert_pretrained
        # # vocab_file_path = "%s-vocab.txt" % self.hparams.bert_pretrained
        # vocab_file_path = "%s-spiece.model" % self.hparams.bert_pretrained
        # # vocab_file_path = "%s-vocab.vocab" % self.hparams.bert_pretrained
        # self._bert_tokenizer = tokenization_albert.AlbertTokenizer(
        #   vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
        #   do_lower_case=False
        # )

        # xlnet 수정
        # bert_pretrained_dir = "/home/heogle/%s/" % self.hparams.bert_pretrained
        # vocab_file_path = "%s-spiece.model" % self.hparams.bert_pretrained
        # self._bert_tokenizer = tokenization_xlnet.XLNetTokenizer(vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path), do_lower_case=False)

        #     electra 수정
        bert_pretrained_dir = "/home/heogle/%s/" % self.hparams.bert_pretrained
        vocab_file_path = "electra-large-discriminator-vocab.txt"
        self._bert_tokenizer = tokenization_electra.ElectraTokenizer(
            vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
            do_lower_case=False
        )

        # Entity Marker : [E1] [/E1] [E2] [/E2]
        if self.hparams.do_entity_marker:
          self._bert_tokenizer.add_tokens(["[E1]","[/E1]","[E2]","[/E2]"])


    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
          self.sentence_id = sentence_id
          self.relation = relation
          self.sentence = sentence
          self.subj_start = subj_ind[0]
          self.subj_end = subj_ind[1]
          self.obj_start = obj_ind[0]
          self.obj_end = obj_ind[1]
        """

        # print("self.input_examples : ", self.input_examples[index].sentence)
        # print("self._annotate_sentence(self.input_examples[index]) : ", self._annotate_sentence(self.input_examples[index]))
        annotated_sentence, entity_markers = self._annotate_sentence(self.input_examples[index])
        current_feature = dict()
        current_feature["relation"] = torch.tensor(self.label2id[self.input_examples[index].relation]).long()
        # print("self.label2id[self.input_examples[index].relation] : ", self.label2id[self.input_examples[index].relation])
        # print("annotated_sentence : ", annotated_sentence)
        current_feature["sentence"] = torch.tensor(annotated_sentence).long()
        current_feature["entity_markers"] = torch.tensor(entity_markers).long()

        return current_feature

    def _annotate_sentence(self, example):

      anno_sent = [tok.lower() for tok in example.sentence]

      # Standard : [CLS], [SEP]
      anno_sent = ["[CLS]"] + anno_sent + ["[SEP]"]
      entity_markers = [example.subj_start+1, example.subj_end+1, example.obj_start+1, example.obj_end+1]

      if self.hparams.do_entity_marker:
        if example.subj_start < example.obj_start:
          anno_sent.insert(example.subj_start + 1, "[E1]")
          anno_sent.insert(example.subj_end + 2, "[/E1]")
          anno_sent.insert(example.obj_start + 3, "[E2]")
          anno_sent.insert(example.obj_end + 4, "[/E2]")
          entity_markers = [example.subj_start+1, example.subj_end+2, example.obj_start+3, example.obj_end+4]

        else:
          anno_sent.insert(example.obj_start + 1, "[E2]")
          anno_sent.insert(example.obj_end + 2, "[/E2]")
          anno_sent.insert(example.subj_start + 3, "[E1]")
          anno_sent.insert(example.subj_end + 4, "[/E1]")
          entity_markers = [example.subj_start + 3, example.subj_end + 4, example.obj_start + 1, example.obj_end + 2]

        #for e_idx in entity_markers:
        #  assert anno_sent[e_idx] in {"[E1]", "[/E1]", "[E2]", "[/E2]"}

      anno_sent = self._bert_tokenizer.convert_tokens_to_ids(anno_sent)
      return anno_sent, entity_markers

    def collate_fn(self, batch):
      merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
      max_sent_len = max([len(sent) for sent in merged_batch["sentence"]])
      for key in merged_batch:
        if key in ["sentence"]:
          for batch_idx, features in enumerate(merged_batch[key]):
            # pad_idx is zero
            pad_features = torch.zeros(max_sent_len - len(features)).long()
            merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)
        merged_batch[key] = torch.stack(merged_batch[key], 0)

      return merged_batch