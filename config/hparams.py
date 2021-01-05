from collections import defaultdict

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_ids = [0],

  # Input params
  train_batch_size=8,
  eval_batch_size=8,
  virtual_batch_size=64,

  # Training params
  learning_rate=3e-5,  # 5e-e, 3e-5, 2e-5, 5e-5
  # learning_rate=1e-5,
  # learning_rate=2e-5,
  # learning_rate=5e-5,
  dropout_keep_prob=0.8,
  num_epochs=10,
  max_gradient_norm=5,

  pad_idx=0,
  max_position_embeddings=100,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  # bert_hidden_dim=768,
  bert_hidden_dim=1024,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,
  # bert
  # bert_pretrained="bert-base-uncased",
  # bert_pretrained="bert-large-uncased",
  bert_pretrained = "electra-large-discriminator",

  # Train Model Config
  task_name="tacred",
  # task_name="semeval",
  # task_name = "fewrel",
  # task_name="kbp",
  do_bert=True,
  do_entity_marker=True,

  # Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
  root_dir="/home/heogle/developing/MTB-RelationExtraction/mtb",
  save_dirpath='checkpoints/',
  data_dir="/home/heogle/developing/MTB-RelationExtraction/data/processed_data/bert/%s_%s.pkl",
  vocab_dir="/home/heogle/developing/MTB-RelationExtraction_taesun/data/news_vocab.txt",
  # bert, albert, xlnet, electra
  # bert_pretrained_dir= "/mnt/raid5/shared/bert/pytorch/%s/",
  bert_pretrained_dir = "/home/heogle/%s/",
  load_pthpath="",

  # nlpserver2 env
  # # Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
  # root_dir="/home/heogle/MTB-RelationExtraction_taesun/mtb/",
  # save_dirpath='checkpoints/',
  # data_dir="/home/heogle/MTB-RelationExtraction_taesun/data/processed_data/%s_%s.pkl",
  # vocab_dir="/home/heogle/MTB-RelationExtraction_taesun/data/news_vocab.txt",
  # bert_pretrained_dir="/mnt/raid5/heogle/LIMIT-BERT/%s/",
  # load_pthpath="",

  cpu_workers=1,
  tensorboard_step=1000,
)

# STANDARD_PARAMS = BASE_PARAMS.copy()
# STANDARD_PARAMS.update(
#   gpu_ids=[0],
#   model_type="standard",
#
#   # Input params
#   train_batch_size=8,
#   eval_batch_size=8,
#   virtual_batch_size=64,
#   # virtual_batch_size=8,
#   do_entity_marker=False,
# )

# bert-large-uncased standard
STANDARD_PARAMS = BASE_PARAMS.copy()
STANDARD_PARAMS.update(
  gpu_ids=[0],

  model_type="standard",
  # Input params
  train_batch_size = 1,
  eval_batch_size = 2,
  virtual_batch_size = 64,
  # virtual_batch_size=32,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
  bert_pretrained="bert-large-uncased",
  bert_hidden_dim=1024,
  do_entity_marker=True,
)

# albert-large-v2 standard
# STANDARD_PARAMS = BASE_PARAMS.copy()
# STANDARD_PARAMS.update(
#   gpu_ids=[0],
#   model_type="standard",
#   # Input params
#   train_batch_size=1,
#   eval_batch_size=2,
#   virtual_batch_size=64,
#   num_hidden_layers=24,
#   num_attention_heads=16,
#   intermediate_size=4096,
#   bert_pretrained="albert-large-v2",
#   albert_hidden_dim=1024,
#   do_entity_marker=False,
# )


MENTION_POOLING_PARAMS = BASE_PARAMS.copy()
MENTION_POOLING_PARAMS.update(
  gpu_ids=[0],
  # Input params
  train_batch_size=1,
  eval_batch_size=2,
  virtual_batch_size=64,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
  bert_pretrained="bert-large-uncased",
  bert_hidden_dim=1024,

  model_type="mention_pooling",
  do_entity_marker=False,
)

# ============= Entity Markers ===============
# bert-large-uncased
# ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
# ENTITY_MARKERS_PARAMS.update(
#   gpu_ids=[0],
#   model_type="entity_markers",
#   # Input params
#   train_batch_size=1,
#   eval_batch_size=2,
#   # virtual_batch_size=64,
#   virtual_batch_size = 32,
#   num_hidden_layers=24,
#   num_attention_heads=16,
#   intermediate_size=4096,
#   bert_pretrained="bert-large-uncased",
#   bert_hidden_dim=1024,
# )

# ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
# ENTITY_MARKERS_PARAMS.update(
#   gpu_ids=[0],
#   # Input params
#   train_batch_size=8,
#   eval_batch_size=8,
#   virtual_batch_size = 64,
#   model_type="entity_markers",
#   do_entity_marker=False,
# )

# albert-large-v2 entity marker 만들기
# ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
# ENTITY_MARKERS_PARAMS.update(
#   gpu_ids=[0],
#   model_type="entity_markers",
#   # Input params
#   train_batch_size=1,
#   eval_batch_size=2,
#   virtual_batch_size=64,
#   num_hidden_layers=24,
#   num_attention_heads=16,
#   intermediate_size=4096,
#   bert_pretrained="albert-large-v2",
#   albert_hidden_dim=1024,
# )


# albert-xxlarge-v2 entity marker 만들기
# ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
# ENTITY_MARKERS_PARAMS.update(
#   gpu_ids=[0],
#   model_type="entity_markers",
#   # Input params
#   train_batch_size=1,
#   eval_batch_size=2,
#   virtual_batch_size=64,
#   num_hidden_layers=12,
#   num_attention_heads=64,
#   intermediate_size=4096,
#   bert_pretrained="albert-xxlarge-v2",
#   albert_hidden_dim=1024,
# )

# xlnet-base
# ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
# ENTITY_MARKERS_PARAMS.update(
#   gpu_ids=[0],
#   # Input params
#   model_type="entity_markers",
#   do_entity_marker=False,
#   bert_pretrained="xlnet-base-cased",
#   vocab_size=32000,
#   d_model=1024,
#   n_layer=24,
#   n_head=16,
#   d_inner=4096,
# )


# electra
ENTITY_MARKERS_PARAMS = BASE_PARAMS.copy()
ENTITY_MARKERS_PARAMS.update(
  gpu_ids=[0],
  # Input params
  model_type="entity_markers",
  bert_pretrained="electra-large-discriminator",
  intermediate_size=4096,
  num_attention_heads = 16,
  num_hidden_layers = 24,
  pad_token_id= 0,
  type_vocab_size = 2,
  vocab_size = 30522
)






























