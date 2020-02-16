# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from app         import db
from flask_login import UserMixin

class User(UserMixin, db.Model):

    id       = db.Column(db.Integer,     primary_key=True)
    user     = db.Column(db.String(64),  unique = True)
    email    = db.Column(db.String(120), unique = True)
    password = db.Column(db.String(500))

    def __init__(self, user, email, password):
        self.user       = user
        self.password   = password
        self.email      = email

    def __repr__(self):
        return str(self.id) + ' - ' + str(self.user)

    def save(self):

        # inject self into db session    
        db.session.add ( self )

        # commit change and save the object
        db.session.commit( )

        return self 
        
import tensorflow as tf
import os
from tensorflow.keras import layers as L
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
from transformers import TFBertMainLayer, TFBertPreTrainedModel, TFRobertaMainLayer, TFRobertaPreTrainedModel
from transformers.modeling_tf_utils import get_initializer
from .helper import get_add_tokens

class TFBertForNaturalQuestionAnswering(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.bert = TFBertMainLayer(config, name='bert')
        self.initializer = get_initializer(config.initializer_range)
        self.qa_outputs = L.Dense(config.num_labels,
            kernel_initializer=self.initializer, name='qa_outputs')
        self.long_outputs = L.Dense(1, kernel_initializer=self.initializer,
            name='long_outputs')

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        long_logits = tf.squeeze(self.long_outputs(sequence_output), -1)
        return start_logits, end_logits, long_logits


class TFRobertaForNaturalQuestionAnswering(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name='roberta')
        self.initializer = get_initializer(config.initializer_range)
        self.qa_outputs = L.Dense(config.num_labels,
            kernel_initializer=self.initializer, name='qa_outputs')
        self.long_outputs = L.Dense(1, kernel_initializer=self.initializer,
            name='long_outputs')

    def call(self, inputs, **kwargs):
        outputs = self.roberta(inputs, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        long_logits = tf.squeeze(self.long_outputs(sequence_output), -1)
        return start_logits, end_logits, long_logits

MODEL_CLASSES = {
    'bert': (BertConfig, TFBertForNaturalQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, TFRobertaForNaturalQuestionAnswering, RobertaTokenizer),
}

model_type = 'bert'
config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
model_config = "./transformers_cache/bert_large_uncased_config.json"

# Set cased / uncased
config_basename = os.path.basename(model_config)
if config_basename.startswith('bert'):
    do_lower_case = 'uncased' in config_basename
elif config_basename.startswith('roberta'):
    # https://github.com/huggingface/transformers/pull/1386/files
    do_lower_case = False

config = config_class.from_json_file(model_config)
vocab_txt = "./transformers_cache/bert_large_uncased_vocab.txt"
tokenizer = tokenizer_class(vocab_txt, do_lower_case=do_lower_case)
do_enumerate = False
tags = get_add_tokens(do_enumerate=do_enumerate)
num_added = tokenizer.add_tokens(tags)
checkpoint_dir = "./nq_bert_uncased_68"
weights_fn = os.path.join(checkpoint_dir, 'weights.h5')
model = model_class(config)
model(model.dummy_inputs, training=False)
model.load_weights(weights_fn)
