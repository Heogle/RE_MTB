import os
import sys
import json
import pickle
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.bert import tokenization_bert
# from models.albert import tokenization_albert
# from models.xlnet import tokenization_xlnet
# from models.electra import tokenization_electra

"""
  TACRED : 68,124
  Relation : 42
"""

class InputExamples(object):
  def __init__(self, sentence_id, sentence, subj_ind, obj_ind, relation=None):

    self.sentence_id = sentence_id
    self.relation = relation
    self.sentence = sentence
    self.subj_start = subj_ind[0]
    self.subj_end = subj_ind[1]
    self.obj_start = obj_ind[0]
    self.obj_end = obj_ind[1]

class TACREDUtils(object):
  def __init__(self, json_path):
    # bert_tokenizer init
    self.json_path = json_path

    self.relation_set = set()
    self._bert_tokenizer_init()

  # bert
  def _bert_tokenizer_init(self, bert_pretrained='bert-large-uncased'):
    bert_pretrained_dir = "/home/heogle/%s/" % bert_pretrained
    vocab_file_path = "%s-vocab.txt" % bert_pretrained

    self._bert_tokenizer = tokenization_bert.BertTokenizer(
      vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path),
      do_lower_case=False
    )
    # print("self._bert_tokenizer : ", self._bert_tokenizer)
    print("BERT tokenizer init completes")

  # albert
  # def _bert_tokenizer_init(self, bert_pretrained='albert-large-v2'):
  #   bert_pretrained_dir = "/home/heogle/%s/" % bert_pretrained
  #   # vocab_file_path = "%s-vocab.vocab" % bert_pretrained
  #   vocab_file_path = "%s-spiece.model" % bert_pretrained
  #
  #   self._bert_tokenizer = tokenization_albert.AlbertTokenizer(vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path), do_lower_case=False)
  #   print("ALBERT tokenizer init completes")

  # XLNET
  # def _bert_tokenizer_init(self, bert_pretrained='xlnet-base-cased'):
  #   bert_pretrained_dir = "/home/heogle/%s/" % bert_pretrained
  #   # vocab_file_path = "%s-vocab.vocab" % bert_pretrained
  #   vocab_file_path = "%s-spiece.model" % bert_pretrained

  #   self._bert_tokenizer = tokenization_xlnet.XLNetTokenizer(vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path), do_lower_case=False)
  #   print("XLNet tokenizer init completes")

  # electra
  # def _bert_tokenizer_init(self, bert_pretrained='electra-large-discriminator'):
  #   bert_pretrained_dir = "/home/heogle/%s/" % bert_pretrained
  #   vocab_file_path = "%s-vocab.txt" % bert_pretrained
  #
  #   self._bert_tokenizer = tokenization_electra.ElectraTokenizer(
  #     vocab_file=os.path.join(bert_pretrained_dir, vocab_file_path), do_lower_case=False)
  #
  #   print("ELECTRA tokenizer init completes")


  def read_json_sample(self, data_type):
    tacred_path = self.json_path % data_type  # train, dev, test
    with open(tacred_path, "r", encoding="utf8") as fr_handle:
      data = json.load(fr_handle)
      # print("(%s) total number of sentence : %d" % (data_type, len(data)))

      for sent_dict in data[0:10]:
        for sent_key in sent_dict.keys():
          print(sent_key, sent_dict[sent_key])
        print("-"*30)


  def read_json(self, data_type):
    print("Loading json file...")
    tacred_path = self.json_path % data_type # train, dev, test
    print("======data type ====== : ", tacred_path)
    with open(tacred_path, "r", encoding="utf8") as fr_handle:
      data = json.load(fr_handle)
      # print("(%s) total number of sentence : %d" % (data_type, len(data)))

    print("====== data ========= : ", data)

    return data

  def make_examples(self, data):
    input_examples = []

    for sent_dict in tqdm(data):
      self.relation_set.add(sent_dict["relation"])

      mod_curr_idx = 0
      entity_markers = [sent_dict["subj_start"], sent_dict["subj_end"], sent_dict["obj_start"], sent_dict["obj_end"]]

      assert len(entity_markers) == 4

      bert_sub_tokens = []
      sub_tok_idx = []
      sub_tok_len = []
      for tok_idx, tok in enumerate(sent_dict["token"]):
        sub_token = self._bert_tokenizer.tokenize(tok)
        bert_sub_tokens.extend(sub_token)

        sub_tok_idx.append(mod_curr_idx)
        mod_curr_idx += len(sub_token)
        sub_tok_len.append(len(sub_token))
    
      if len(bert_sub_tokens) > 512:
        pass
      else:

        subj_ind = (sub_tok_idx[entity_markers[0]],sub_tok_idx[entity_markers[1]] + sub_tok_len[entity_markers[1]])
        obj_ind = (sub_tok_idx[entity_markers[2]], sub_tok_idx[entity_markers[3]] + sub_tok_len[entity_markers[3]])

        # print(bert_sub_tokens[sub_tok_idx[entity_markers[0]]:sub_tok_idx[entity_markers[1]]+sub_tok_len[entity_markers[1]]])
        # print(bert_sub_tokens[sub_tok_idx[entity_markers[2]]:sub_tok_idx[entity_markers[3]]+sub_tok_len[entity_markers[3]]])

        input_examples.append(InputExamples(
          sentence_id=sent_dict["id"],
          sentence=bert_sub_tokens,
          subj_ind=subj_ind, obj_ind=obj_ind,
          relation = sent_dict["relation"]
        ))

    return input_examples

  def make_pkl(self, examples, tacred_pkl_path):
    print("examples : ", examples)
    with open(tacred_pkl_path, "wb") as pkl_handle:
      pickle.dump(examples, pkl_handle)
    print(tacred_pkl_path, " save completes!")

  def read_pkl(self, tacred_pkl_path):
    with open(tacred_pkl_path, "rb") as pkl_handle:
      examples = pickle.load(pkl_handle)
    print(tacred_pkl_path, " save completes!")
    print(len(examples))
    file_name = data_type + ".txt"
    file_txt = open(file_name, "w")
    relation = []
    for e in examples:
      # print(e.sentence_id)
      # print(e.sentence)
      file_txt.write(e.sentence_id + "\n")
      file_txt.write(str(e.sentence) + "\n")

      sentence = self._bert_tokenizer.convert_ids_to_tokens(e.sentence)
      # print(sentence[e.subj_start:e.subj_end])
      # print(sentence[e.obj_start:e.obj_end])
      # print(e.relation)
      # print("-"*200)
      relation.append(e.relation)

      file_txt.write(str(sentence[e.subj_start:e.subj_end]) + "\n")
      file_txt.write(str(sentence[e.obj_start:e.obj_end]) + "\n")
      file_txt.write(e.relation + "\n")
      file_txt.write("-"*200 + "\n")
    print(len(set(relation)))

if __name__ == '__main__':
  # tacred
  tacred_json_path = "/home/heogle/developing/MTB-RelationExtraction/data/tacred_LDC2018T24/data/json/%s.json"
  tacred_pkl_path = "/home/heogle/developing/MTB-RelationExtraction/data/processed_data/bert/tacred_%s.pkl"


  tacred_utils = TACREDUtils(tacred_pkl_path)
  tacred_utils = TACREDUtils(tacred_json_path)
  for data_type in ["test", "dev", "train"]: #"dev",
    data = tacred_utils.read_json(data_type)
    examples = tacred_utils.make_examples(data)
    tacred_utils.make_pkl(examples, tacred_pkl_path % data_type)
    # tacred_utils.read_pkl(tacred_pkl_path % data_type)
