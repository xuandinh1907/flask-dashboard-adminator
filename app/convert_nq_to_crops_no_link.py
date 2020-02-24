from .models import model,tokenizer,do_enumerate,do_lower_case
from .helper import get_add_tokens
import collections
import os
import bs4
import requests
import time
import tensorflow as tf
import numpy as np
import re

eval_batch_size = 1
UNMAPPED = -123

NQExample = collections.namedtuple("NQExample", [
    "qas_id", "question_text", "doc_tokens","crop_start"])

Crop = collections.namedtuple("Crop", ["example_id","unique_id", "doc_span_index",
    "tokens", "token_to_orig_map", "token_is_max_context",
    "input_ids", "attention_mask", "token_type_ids",
    "paragraph_len"])

DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits","long_logits"])

PrelimPrediction = collections.namedtuple("PrelimPrediction",
    ["crop_index", "start_index", "end_index", "start_logit", "end_logit"])

NbestPrediction = collections.namedtuple("NbestPrediction", [
    "text", "start_logit", "end_logit",
    "start_index", "end_index",
    "orig_doc_start", "orig_doc_end", "crop_index"])

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_spans(doc_stride, max_tokens_for_doc, max_len):
    doc_spans = []
    start_offset = 0
    while start_offset < max_len:
        length = max_len - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(DocSpan(start=start_offset, length=length))
        if start_offset + length == max_len:
            break
        start_offset += min(length, doc_stride)
    return doc_spans

def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_crops(example, tokenizer,UNMAPPED=-123, max_seq_length=512,
                              doc_stride=256, max_query_length=64, is_training=False,
                              cls_token='[CLS]', sep_token='[SEP]', pad_id=0,
                              sequence_a_segment_id=0,
                              sequence_b_segment_id=1,
                              cls_token_segment_id=0,
                              pad_token_segment_id=0,
                              mask_padding_with_zero=True,
                              sep_token_extra=False):
    """Loads an example into a list of `InputBatch`s."""
    unique_id = 1000000000
    sub_token_cache = {}
    
    crops = []

    query_tokens = tokenizer.tokenize(example.question_text)
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    # this takes the longest!
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = sub_token_cache.get(token)
        if sub_tokens is None:
            sub_tokens = tokenizer.tokenize(token)
            sub_token_cache[token] = sub_tokens
        tok_to_orig_index.extend([i for _ in range(len(sub_tokens))])
        all_doc_tokens.extend(sub_tokens)

    tok_start_position = None
    tok_end_position = None

    # For Bert: [CLS] question [SEP] paragraph [SEP]
    special_tokens_count = 3
    max_tokens_for_doc = max_seq_length - len(query_tokens) - special_tokens_count
    assert max_tokens_for_doc > 0
    # We can have documents that are longer than the maximum
    # sequence length. To deal with this we do a sliding window
    # approach, where we take chunks of the up to our max length
    # with a stride of `doc_stride`.
    doc_spans = get_spans(doc_stride, max_tokens_for_doc, len(all_doc_tokens))
    for doc_span_index, doc_span in enumerate(doc_spans):
        # Tokens are constructed as: CLS Query SEP Paragraph SEP
        tokens = []
        token_to_orig_map = UNMAPPED * np.ones((max_seq_length, ), dtype=np.int32)
        token_is_max_context = np.zeros((max_seq_length, ), dtype=np.bool)
        token_type_ids = []
        special_tokens_offset = special_tokens_count - 1
        doc_offset = len(query_tokens) + special_tokens_offset

        # CLS token at the beginning
        tokens.append(cls_token)
        token_type_ids.append(cls_token_segment_id)

        # Query
        tokens += query_tokens
        token_type_ids += [sequence_a_segment_id] * len(query_tokens)

        # SEP token
        tokens.append(sep_token)
        token_type_ids.append(sequence_a_segment_id)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            # We add `example.crop_start` as the original document
            # is already shifted
            token_to_orig_map[len(tokens)] = tok_to_orig_index[
                split_token_index] + example.crop_start

            token_is_max_context[len(tokens)] = check_is_max_context(doc_spans,
                doc_span_index, split_token_index)
            tokens.append(all_doc_tokens[split_token_index])
            token_type_ids.append(sequence_b_segment_id)

        paragraph_len = doc_span.length

        # SEP token
        tokens.append(sep_token)
        token_type_ids.append(sequence_b_segment_id)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_id)
            attention_mask.append(0 if mask_padding_with_zero else 1)
            token_type_ids.append(pad_token_segment_id)

        # reduce memory, only input_ids needs more bits
        input_ids = np.array(input_ids, dtype=np.int32)
        attention_mask = np.array(attention_mask, dtype=np.bool)
        token_type_ids = np.array(token_type_ids, dtype=np.uint8)

        crop = Crop(
            example_id = example.qas_id,
            unique_id=unique_id,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            paragraph_len=paragraph_len)
        crops.append(crop)
        unique_id += 1

    return crops

@tf.function
def predict_step(batch):
    outputs = model(batch, training=False)
    return outputs

def prelim_predict(example_id,part_of_crops,unique_id_to_result,n_best_size = 10,max_answer_length = 30,UNMAPPED = -123):
    short_prelim_predictions = []
    for crop_index, crop in enumerate(part_of_crops):
        result = unique_id_to_result[crop.unique_id]
        start_indexes = np.argpartition(result.start_logits, -n_best_size)[-n_best_size:]
        start_indexes = [int(x) for x in start_indexes]
        end_indexes = np.argpartition(result.end_logits, -n_best_size)[-n_best_size:]
        end_indexes = [int(x) for x in end_indexes]

        # create short answers
        for start_index in start_indexes:
            if start_index >= len(crop.tokens):
                continue
            if crop.token_to_orig_map[start_index] == UNMAPPED:
                continue
            if not crop.token_is_max_context[start_index]:
                continue

            for end_index in end_indexes:
                if end_index >= len(crop.tokens):
                    continue
                if crop.token_to_orig_map[end_index] == UNMAPPED:
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue

                short_prelim_predictions.append(PrelimPrediction(
                    crop_index=crop_index,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))

    short_prelim_predictions = sorted(short_prelim_predictions,
            key=lambda x: x.start_logit + x.end_logit, reverse=True)
    return short_prelim_predictions

def get_nbest(prelim_predictions, crops, example, n_best_size=10):
    seen, nbest = set(), []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        crop = crops[pred.crop_index]
        orig_doc_start, orig_doc_end = -1, -1
        # non-null
        if pred.start_index > 0:
            # Long answer has no end_index. We still generate some text to check
            if pred.end_index == -1:
                tok_tokens = crop.tokens[pred.start_index: pred.start_index + 11]
            else:
                tok_tokens = crop.tokens[pred.start_index: pred.end_index + 1]
            tok_text = " ".join(tok_tokens)
            tok_text = clean_text(tok_text)

            orig_doc_start = int(crop.token_to_orig_map[pred.start_index])
            if pred.end_index == -1:
                orig_doc_end = orig_doc_start + 10
            else:
                orig_doc_end = int(crop.token_to_orig_map[pred.end_index])

            final_text = tok_text
            if final_text in seen:
                continue

        else:
            final_text = ""

        seen.add(final_text)
        nbest.append(NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit, end_logit=pred.end_logit,
            start_index=pred.start_index, end_index=pred.end_index,
            orig_doc_start=orig_doc_start, orig_doc_end=orig_doc_end,
            crop_index=pred.crop_index))

    # Degenerate case. I never saw this happen.
    if len(nbest) in (0, 1):
        nbest.insert(0, NbestPrediction(text="empty",
            start_logit=0.0, end_logit=0.0,
            start_index=-1, end_index=-1,
            orig_doc_start=-1, orig_doc_end=-1,
            crop_index=UNMAPPED))

    assert len(nbest) >= 1
    return nbest

def clean_text(tok_text):
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text

def get_doc_tokens(paragraph_text):
    doc_tokens = []
    prev_is_whitespace = True
    for c in paragraph_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens




def demo_no_link(document_text,questions) :
    squad = collections.defaultdict(list)
    # document_text = get_document_text(url)
    # print(document_text)
    doc_tokens = get_doc_tokens(document_text)
    # print(doc_tokens)
    for example_id,question in enumerate(questions) :
        tic = time.time()
        
        ## build NQExample
        qa = {'question': question, 'id': example_id, 'crop_start': 0}
        example = NQExample(
                qas_id=qa["id"],
                question_text=qa["question"],
                doc_tokens=doc_tokens,
                crop_start=qa["crop_start"])
        # print(example)
        ## compute crops
        crops = convert_examples_to_crops(example, tokenizer)
        # print(crops)
        ## build dataset
        all_input_ids = tf.stack([c.input_ids for c in crops], 0)
        all_attention_mask = tf.stack([c.attention_mask for c in crops], 0)
        all_token_type_ids = tf.stack([c.token_type_ids for c in crops], 0)
        dataset = [all_input_ids, all_attention_mask, all_token_type_ids]
        eval_ds = tf.data.Dataset.from_tensor_slices({
        'input_ids': tf.constant(dataset[0]),
        'attention_mask': tf.constant(dataset[1]),
        'token_type_ids': tf.constant(dataset[2]),
        'example_index': tf.range(len(dataset[0]), dtype=tf.int32)
        })
        eval_ds = eval_ds.batch(batch_size=eval_batch_size, drop_remainder=True)
        
        ## raw results making
        all_results = []
        for batch in eval_ds:
            example_indexes = batch['example_index']
            outputs = predict_step(batch)
            batched_start_logits = outputs[0].numpy()
            batched_end_logits = outputs[1].numpy()
            batched_long_logits = outputs[2].numpy()
            for i, example_index in enumerate(example_indexes):

                eval_crop = crops[example_index]
                unique_id = int(eval_crop.unique_id)
                start_logits = batched_start_logits[i].tolist()
                end_logits = batched_end_logits[i].tolist()
                long_logits = batched_long_logits[i].tolist()

                result = RawResult(unique_id=unique_id,
                                start_logits=start_logits,
                                end_logits=end_logits,
                                long_logits=long_logits)
                all_results.append(result)
        
        ## premilinary predictions
        example_index_to_crops = collections.defaultdict(list)
        for crop in crops :
            example_index_to_crops[crop.example_id].append(crop)
        unique_id_to_result = {result.unique_id: result for result in all_results}
        # all_predictions = collections.OrderedDict()
        part_of_crops = example_index_to_crops[example_id]
        short_prelim_predictions = prelim_predict(example_id,part_of_crops,unique_id_to_result)
        short_nbest = get_nbest(short_prelim_predictions, part_of_crops,example)
        # print(short_nbest)
        ## Show results
        short_best_non_null = short_nbest[0].text
        for entry in short_nbest[1:]:
            if len(entry.text) > len(short_best_non_null) and short_best_non_null in entry.text:
                    short_best_non_null = " ".join(doc_tokens[entry.orig_doc_start:entry.orig_doc_end])
                    short_best_non_null = re.sub('[^a-zA-Z0-9 ]','', short_best_non_null).strip().lower()

        print(short_best_non_null)
        text = list(map(lambda x: x.replace('\n',''), document_text.split('\n\n')))
        # print(text)
        for i, line in enumerate(text):
            
            regex_line = re.sub('[^a-zA-Z0-9 ]','', line).strip().lower()
            if short_best_non_null in regex_line:
                
                print(regex_line)
                squad[question].append(line)
        squad[question].append("Finding answer time "+str(round(time.time() - tic,1))+" s")
    
    return squad
    # print("Quesion :",question)
    # print("Answer :",short_best_non_null)
    # print("Finding answer time :",round(time.time() - tic,1),"s")
    # print("-----------------------------------------------")