import os
import sys
import csv
import json
import torch
import pickle
import logging
import inspect
import random
random.seed(1)
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

logger = logging.getLogger(__name__)


def lmap(a, b):
    return list(map(a,b))


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True
            
            model_name = my_args.model_name_or_path.split("/")[-1]
            is_pretrain = my_args.pretrain
            cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}_pretrain{is_pretrain}.pkl")
            refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


# def solve(line,  set_type="train", pretrain=1):
#     """_summary_

#     Args:
#         line (_type_): (head, relation, tail), wiki q_id, if pretrain=1: head = tail, relation = rel[0]
#         set_type (str, optional): _description_. Defaults to "train".
#         pretrain (int, optional): _description_. Defaults to 1.

#     Returns:
#         _type_: _description_
#     """
#     examples = []

#     guid = "%s-%s" % (set_type, 0)
    
#     if pretrain:
#         head, rel, tail = line
#         relation_text = rel2text[rel]
        
#         rnd = random.random()
#         if rnd <= 0.4:
#             # (T, T) -> (head, rel, MASK)
#             head_ent_text = ent2text[head]
#             tail_ent_text = ent2text[tail]
#             head_ent = None
#             tail_ent = None
            
#         elif rnd > 0.4 and rnd < 0.7:
#             # (I, T)
#             head_ent_text = ""
#             tail_ent_text = ent2text[tail]
#             head_ent = head
#             tail_ent = None
#         else:
#             # (I, I)
#             head_ent_text = ""
#             tail_ent_text = ""
#             head_ent = head
#             tail_ent = tail
            
#         struct_head_id = ent2struct_id.get(head, -1)
#         struct_tail_id = ent2struct_id.get(tail, -1)
#         struct_rel_id = num_struct_entities + rel2struct_id.get(rel, -1) if rel in rel2struct_id else -1
        
#         # (head, rel, MASK)
#         examples.append(
#                 InputExample(
#                     guid=guid,
#                     text_a="[UNK] " + head_ent_text, 
#                     text_b="[PAD] " + relation_text,
#                     text_c="[MASK]", 
#                     real_label=ent2id[tail],
#                     head_id=ent2id[head], 
#                     head_ent=head_ent, 
#                     pre_type=1,
#                     struct_head_id=struct_head_id,
#                     struct_rel_id=struct_rel_id,
#                     struct_tail_id=struct_tail_id,
#                 )
#             )
#         # (head, MASK, tail)
#         examples.append(
#             InputExample(
#                 guid=guid,
#                 text_a="[UNK] " + head_ent_text, 
#                 text_b="[MASK]",
#                 text_c="[UNK] " + tail_ent_text, 
#                 real_label=rel2id[rel],
#                 head_id=ent2id[head], 
#                 head_ent=head_ent, 
#                 tail_ent=tail_ent,
#                 pre_type=2,
#                 struct_head_id=struct_head_id,
#                 struct_rel_id=struct_rel_id,
#                 struct_tail_id=struct_tail_id,
#             )
#         )
#     else:
#         head, rel, tail = line['example'][0], line['relation'], line['example'][1]
#         question, answer = line['question'], line['answer']
#         mode = line['mode']
#         # +++ 添加此代码块以获取所有相关实体的结构化ID +++
#         # q_head 和 q_tail 对应原始三元组的 head 和 tail
#         struct_q_head_id = ent2struct_id.get(head, -1)
#         struct_q_tail_id = ent2struct_id.get(tail, -1)
#         # a_head 对应 analogy 任务中的 question
#         struct_a_head_id = ent2struct_id.get(question, -1)
#         struct_rel_id = num_struct_entities + rel2struct_id.get(rel, -1) if rel in rel2struct_id else -1
        
#         if mode == 0:
#             head_ent_text, tail_ent_text = ent2text[head], ent2text[tail]
#             # (T1, T2) -> (I1, ?)
#             examples.append(
#                 AnalogyInputExample(
#                     guid=guid, 
#                     text_a="[UNK] " + head_ent_text, 
#                     text_b="[PAD]",
#                     text_c="[UNK] " + tail_ent_text, 
#                     text_d="[UNK] ",
#                     text_e="[PAD]",
#                     text_f="[MASK]",
#                     real_label=analogy_ent2id[answer], 
#                     relation=rel2id[rel],
#                     q_head_id=ent2id[head], 
#                     q_tail_id=ent2id[tail], 
#                     a_head_id=ent2id[question],
#                     head_ent=question, 
#                     tail_ent=None,
#                     # +++ 传入新的ID +++
#                     struct_q_head_id=struct_q_head_id,
#                     struct_q_tail_id=struct_q_tail_id,
#                     struct_a_head_id=struct_a_head_id,
#                     struct_rel_id=struct_rel_id
#                 )
#             )
#         elif mode == 1:
#             head_ent_text = ent2text[question]
#             # (I1, I2) -> (T1, ?)
#             examples.append(
#                 AnalogyInputExample(
#                     guid=guid, 
#                     text_a="[UNK] ", 
#                     text_b="[PAD]", 
#                     text_c="[UNK] ", 
#                     text_d="[UNK] " + head_ent_text,
#                     text_e="[PAD]", 
#                     text_f="[MASK]",
#                     real_label=analogy_ent2id[answer], 
#                     relation=rel2id[rel],
#                     q_head_id=ent2id[head], 
#                     q_tail_id=ent2id[tail], 
#                     a_head_id=ent2id[question],
#                     head_ent=head, 
#                     tail_ent=tail,
#                     # +++ 传入新的ID +++
#                     struct_q_head_id=struct_q_head_id,
#                     struct_q_tail_id=struct_q_tail_id,
#                     struct_a_head_id=struct_a_head_id,
#                     struct_rel_id=struct_rel_id
#                 )
#             )
#         elif mode == 2:
#             tail_ent_text = ent2text[tail]
#             # (I1, T1) -> (I2, ?)
#             examples.append(
#                 AnalogyInputExample(
#                     guid=guid, 
#                     text_a="[UNK] ", 
#                     text_b="[PAD]", 
#                     text_c="[UNK] " + tail_ent_text, 
#                     text_d="[UNK] ",
#                     text_e="[PAD]", 
#                     text_f="[MASK]",
#                     real_label=analogy_ent2id[answer], 
#                     relation=rel2id[rel],
#                     q_head_id=ent2id[head], 
#                     q_tail_id=ent2id[tail], 
#                     a_head_id=ent2id[question],
#                     head_ent=head, 
#                     tail_ent=question,
#                     # +++ 传入新的ID +++
#                     struct_q_head_id=struct_q_head_id,
#                     struct_q_tail_id=struct_q_tail_id,
#                     struct_a_head_id=struct_a_head_id,
#                     struct_rel_id=struct_rel_id
#                 )
#             )
#     return examples


def solve(line,  set_type="train", pretrain=1):
    """_summary_

    Args:
        line (_type_): (head, relation, tail), wiki q_id, if pretrain=1: head = tail, relation = rel[0]
        set_type (str, optional): _description_. Defaults to "train".
        pretrain (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    examples = []
    guid = "%s-%s" % (set_type, 0)
    
    # +++ ADDED +++
    # Set a max character length for descriptions to prevent truncation errors with long text.
    # This leaves enough space for other tokens like [MASK], [SEP], etc.
    MAX_DESC_LEN = 120

    if pretrain:
        head, rel, tail = line
        # Use .get() for safety to avoid KeyError if a relation is not in the map
        relation_text = rel2text.get(rel, "")
        
        rnd = random.random()
        if rnd <= 0.4:
            # (T, T) -> (head, rel, MASK)
            # Use .get() for safety to avoid KeyError
            head_ent_text = ent2text.get(head, "")
            tail_ent_text = ent2text.get(tail, "")

            # +++ ADDED +++ Truncate long descriptions
            if len(head_ent_text) > MAX_DESC_LEN:
                head_ent_text = head_ent_text[:MAX_DESC_LEN]
            if len(tail_ent_text) > MAX_DESC_LEN:
                tail_ent_text = tail_ent_text[:MAX_DESC_LEN]
            
            head_ent = None
            tail_ent = None
            
        elif rnd > 0.4 and rnd < 0.7:
            # (I, T)
            head_ent_text = ""
            tail_ent_text = ent2text.get(tail, "")

            # +++ ADDED +++ Truncate long descriptions
            if len(tail_ent_text) > MAX_DESC_LEN:
                tail_ent_text = tail_ent_text[:MAX_DESC_LEN]
                
            head_ent = head
            tail_ent = None
        else:
            # (I, I)
            head_ent_text = ""
            tail_ent_text = ""
            head_ent = head
            tail_ent = tail
            
        struct_head_id = ent2struct_id.get(head, -1)
        struct_tail_id = ent2struct_id.get(tail, -1)
        struct_rel_id = num_struct_entities + rel2struct_id.get(rel, -1) if rel in rel2struct_id else -1
        
        # (head, rel, MASK)
        examples.append(
                InputExample(
                    guid=guid,
                    text_a="[UNK] " + head_ent_text, 
                    text_b="[PAD] " + relation_text,
                    text_c="[MASK]", 
                    real_label=ent2id.get(tail, -1),
                    head_id=ent2id.get(head, -1), 
                    head_ent=head_ent, 
                    pre_type=1,
                    struct_head_id=struct_head_id,
                    struct_rel_id=struct_rel_id,
                    struct_tail_id=struct_tail_id,
                )
            )
        # (head, MASK, tail)
        examples.append(
            InputExample(
                guid=guid,
                text_a="[UNK] " + head_ent_text, 
                text_b="[MASK]",
                text_c="[UNK] " + tail_ent_text, 
                real_label=rel2id.get(rel, -1),
                head_id=ent2id.get(head, -1), 
                head_ent=head_ent, 
                tail_ent=tail_ent,
                pre_type=2,
                struct_head_id=struct_head_id,
                struct_rel_id=struct_rel_id,
                struct_tail_id=struct_tail_id,
            )
        )
    else:
        # This is the fine-tuning branch
        head, rel, tail = line['example'][0], line['relation'], line['example'][1]
        question, answer = line['question'], line['answer']
        mode = line['mode']
        
        # +++ Get all relevant struct IDs +++
        # q_head and q_tail correspond to the head and tail of the original triple
        struct_q_head_id = ent2struct_id.get(head, -1)
        struct_q_tail_id = ent2struct_id.get(tail, -1)
        # a_head corresponds to the question in the analogy task
        struct_a_head_id = ent2struct_id.get(question, -1)
        struct_rel_id = num_struct_entities + rel2struct_id.get(rel, -1) if rel in rel2struct_id else -1
        
        if mode == 0:
            head_ent_text = ent2text.get(head, "")
            tail_ent_text = ent2text.get(tail, "")
            
            # +++ ADDED +++ Truncate long descriptions
            if len(head_ent_text) > MAX_DESC_LEN:
                head_ent_text = head_ent_text[:MAX_DESC_LEN]
            if len(tail_ent_text) > MAX_DESC_LEN:
                tail_ent_text = tail_ent_text[:MAX_DESC_LEN]

            # (T1, T2) -> (I1, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] " + head_ent_text, 
                    text_b="[PAD]",
                    text_c="[UNK] " + tail_ent_text, 
                    text_d="[UNK] ",
                    text_e="[PAD]",
                    text_f="[MASK]",
                    real_label=analogy_ent2id.get(answer, -1), 
                    relation=rel2id.get(rel, -1),
                    q_head_id=ent2id.get(head, -1), 
                    q_tail_id=ent2id.get(tail, -1), 
                    a_head_id=ent2id.get(question, -1),
                    head_ent=question, 
                    tail_ent=None,
                    # +++ Pass new IDs +++
                    struct_q_head_id=struct_q_head_id,
                    struct_q_tail_id=struct_q_tail_id,
                    struct_a_head_id=struct_a_head_id,
                    struct_rel_id=struct_rel_id
                )
            )
        elif mode == 1:
            head_ent_text = ent2text.get(question, "")

            # +++ ADDED +++ Truncate long descriptions
            if len(head_ent_text) > MAX_DESC_LEN:
                head_ent_text = head_ent_text[:MAX_DESC_LEN]

            # (I1, I2) -> (T1, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] ", 
                    text_b="[PAD]", 
                    text_c="[UNK] ", 
                    text_d="[UNK] " + head_ent_text,
                    text_e="[PAD]", 
                    text_f="[MASK]",
                    real_label=analogy_ent2id.get(answer, -1), 
                    relation=rel2id.get(rel, -1),
                    q_head_id=ent2id.get(head, -1), 
                    q_tail_id=ent2id.get(tail, -1), 
                    a_head_id=ent2id.get(question, -1),
                    head_ent=head, 
                    tail_ent=tail,
                    # +++ Pass new IDs +++
                    struct_q_head_id=struct_q_head_id,
                    struct_q_tail_id=struct_q_tail_id,
                    struct_a_head_id=struct_a_head_id,
                    struct_rel_id=struct_rel_id
                )
            )
        elif mode == 2:
            tail_ent_text = ent2text.get(tail, "")

            # +++ ADDED +++ Truncate long descriptions
            if len(tail_ent_text) > MAX_DESC_LEN:
                tail_ent_text = tail_ent_text[:MAX_DESC_LEN]

            # (I1, T1) -> (I2, ?)
            examples.append(
                AnalogyInputExample(
                    guid=guid, 
                    text_a="[UNK] ", 
                    text_b="[PAD]", 
                    text_c="[UNK] " + tail_ent_text, 
                    text_d="[UNK] ",
                    text_e="[PAD]", 
                    text_f="[MASK]",
                    real_label=analogy_ent2id.get(answer, -1), 
                    relation=rel2id.get(rel, -1),
                    q_head_id=ent2id.get(head, -1), 
                    q_tail_id=ent2id.get(tail, -1), 
                    a_head_id=ent2id.get(question, -1),
                    head_ent=head, 
                    tail_ent=question,
                    # +++ Pass new IDs +++
                    struct_q_head_id=struct_q_head_id,
                    struct_q_tail_id=struct_q_tail_id,
                    struct_a_head_id=struct_a_head_id,
                    struct_rel_id=struct_rel_id
                )
            )
    return examples


def filter_init(t1,t2, ent2id_, ent2token_, rel2id_, analogy_ent2id_, analogy_rel2id_,ent2struct_id_, rel2struct_id_, num_struct_entities_):
    global ent2text
    global rel2text
    global ent2id
    global ent2token
    global rel2id
    global analogy_ent2id
    global analogy_rel2id
    global ent2struct_id
    global rel2struct_id
    global num_struct_entities

    ent2text =t1
    rel2text =t2
    ent2id = ent2id_
    ent2token = ent2token_
    rel2id = rel2id_
    analogy_ent2id = analogy_ent2id_
    analogy_rel2id = analogy_rel2id_
    # +++ 赋值给新的全局变量 +++
    ent2struct_id = ent2struct_id_
    rel2struct_id = rel2struct_id_
    num_struct_entities = num_struct_entities_

def delete_init(ent2text_):
    global ent2text
    ent2text = ent2text_


@cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, mode):

    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    
    features = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    encoder = MultiprocessingEncoder(tokenizer, args)
    pool = Pool(16, initializer=encoder.initializer)
    encoder.initializer()
    encoded_lines = pool.imap(encoder.encode_lines, examples, 1000)

    for enc_lines in tqdm(encoded_lines, total=len(examples)):
        for enc_line in enc_lines:
            features.append(enc_line)

    num_entities = len(processor.get_entities(args.data_dir))
    num_relations = len(processor.get_relations(args.data_dir))
    if args.pretrain:
        for f_id, feature in enumerate(features):
            head_id, rel_id, tail_id = feature.pop('head_id'), feature.pop('rel_id'), feature.pop('tail_id')
            if head_id != -1 and  tail_id != -1:
                # relation prediction
                count = 0
                entity_id = [head_id, tail_id]
                for i, ids in enumerate(feature['input_ids']):
                    if ids == tokenizer.unk_token_id and count < 2:
                        features[f_id]['input_ids'][i] = entity_id[count] + len(tokenizer)
                        count += 1
            else:
                # link prediction
                entity_id = head_id if head_id != -1 else tail_id
                for i, ids in enumerate(feature['input_ids']):
                    if ids == tokenizer.unk_token_id:
                        features[f_id]['input_ids'][i] = entity_id + len(tokenizer)
                        break
            
            # for i, ids in enumerate(feature['input_ids']):
            #     if ids == tokenizer.pad_token_id:
            #         features[f_id]['input_ids'][i] = rel_id + len(tokenizer) + num_entities
            #         break
    else:
        for f_id, feature in enumerate(features):
            q_head_id, q_tail_id, a_head_id = feature.pop('q_head_id'), feature.pop('q_tail_id'), feature.pop('a_head_id')
            count = 0
            entity_id = [q_head_id, q_tail_id, a_head_id]
            rel_idx, sep_idx = [], []
            for i, ids in enumerate(feature['input_ids']):
                if count < 3 and ids == tokenizer.unk_token_id:
                    features[f_id]['input_ids'][i] = entity_id[count] + len(tokenizer)
                    if count == 0:
                        q_head_idx = i
                    elif count == 1: # +++ 新增对 q_tail_idx 的处理 +++
                        q_tail_idx = i
                    elif count == 2:
                        a_head_idx = i
                    count += 1
                if ids == tokenizer.sep_token_id:
                    sep_idx.append(i)

            features[f_id]['sep_idx'] = sep_idx
            features[f_id]['q_head_idx'] = q_head_idx
            features[f_id]['q_tail_idx'] = q_tail_idx # +++ 新增赋值 +++
            features[f_id]['a_head_idx'] = a_head_idx
            rel_id = features[f_id]['rel_label']
            
            for i, ids in enumerate(feature['input_ids']):
                if ids == tokenizer.pad_token_id:
                    features[f_id]['input_ids'][i] = len(tokenizer) + num_entities + num_relations
                    rel_idx.append(i)
                
            features[f_id]['rel_idx'] = rel_idx
    features = KGCDataset(features)
    return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
            self,
            guid, 
            text_a, 
            text_b=None, 
            text_c=None, 
            pre_type=0, 
            real_label=None, 
            head_id=-1, 
            rel_id=-1,
            tail_id=-1, 
            head_ent=None,
            tail_ent=None,
            # change, add struct embedding
            struct_head_id=None,
            struct_rel_id=None,
            struct_tail_id=None
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            pre_type: The type of the pretrained example, 0 is (des, is, MASK) , 1 is (head, rel, MASK), 2 is (head, MASK, tail)
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
           
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.pre_type = pre_type
        self.real_label = real_label
        self.head_id = head_id
        self.rel_id = rel_id    # rel id
        self.tail_id = tail_id
        self.head_ent = head_ent
        self.tail_ent = tail_ent
        # change, add struct embedding
        self.struct_head_id = struct_head_id
        self.struct_rel_id = struct_rel_id
        self.struct_tail_id = struct_tail_id


class AnalogyInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
            self,
            guid, 
            text_a, 
            text_b=None, 
            text_c=None, 
            text_d=None,
            text_e=None,
            text_f=None,
            real_label=None,
            relation=None,
            q_head_id=-1, 
            q_tail_id=-1, 
            a_head_id=-1,
            head_ent=None,
            tail_ent=None,
            struct_q_head_id=None,
            struct_q_tail_id=None,
            struct_a_head_id=None,
            struct_rel_id=None
            
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            pre_type: The type of the pretrained example, 0 is (des, is, MASK) , 1 is (head, rel, MASK), 2 is (head, MASK, tail)
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
           
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.text_e = text_e
        self.text_f = text_f
        self.real_label = real_label
        self.relation = relation
        self.q_head_id = q_head_id       # entity id
        self.q_tail_id = q_tail_id
        self.a_head_id = a_head_id
        self.head_ent = head_ent     # entity des
        self.tail_ent = tail_ent
        self.struct_q_head_id = struct_q_head_id
        self.struct_q_tail_id = struct_q_tail_id
        self.struct_a_head_id = struct_a_head_id
        self.struct_rel_id = struct_rel_id


@dataclass
class InputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor = None
    head_id: torch.Tensor = -1
    rel_id: torch.Tensor = -1
    tail_id: torch.Tensor = -1
    head_ent: torch.Tensor = None
    tail_ent: torch.Tensor = None
    pre_type: torch.Tensor = None
    token_type_ids: torch.Tensor = None
    
    # change, add struct embedding
    struct_head_id: int = None
    struct_rel_id: int = None
    struct_tail_id: int = None
    
    head_token_idx: int = -1
    tail_token_idx: int = -1

@dataclass
class AnalogyInputFeatures:
    """A single set of features of data."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    label: torch.Tensor = None
    rel_label: torch.Tensor = None
    q_head_id: torch.Tensor = -1
    q_tail_id: torch.Tensor = -1
    a_head_id: torch.Tensor = -1
    head_ent: torch.Tensor = None
    tail_ent: torch.Tensor = None
    # +++ 添加所有这些新字段 +++
    struct_q_head_id: int = -1
    struct_q_tail_id: int = -1
    struct_a_head_id: int = -1
    struct_rel_id: int = -1
    q_head_token_idx: int = -1
    q_tail_token_idx: int = -1
    a_head_token_idx: int = -1


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    @classmethod
    def _read_txt(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        read_lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                head, rel, tail = line.split(quotechar)
                read_lines.append((head, rel, tail.replace('\n', '')))
        return read_lines

    @classmethod
    def _read_dict_txt(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        read_dict = {}
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split(quotechar)
                read_dict[key] = value[:-1]
        return read_dict

    @classmethod
    def _read_json(cls, input_file, quotechar='\t'):
        """Reads a `quotechar` separated txt file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            read_lines = [json.loads(line) for line in lines]
        return read_lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self, tokenizer, args, struct_entity2id, struct_relation2id):
        self.labels = set()
        self.tokenizer = tokenizer
        self.args = args
        #change, add struct embedding
        self.struct_entity2id = struct_entity2id
        self.struct_relation2id = struct_relation2id
        self.num_struct_entities = len(struct_entity2id)
        
        self.entity_path = os.path.join(self.args.pretrain_path, "entity2textlong.txt") if os.path.exists(os.path.join(self.args.pretrain_path, 'entity2textlong.txt')) \
        else os.path.join(self.args.pretrain_path, "entity2text.txt")

    def get_train_examples(self, data_dir):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "train", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "train.json")), "train", data_dir, self.args)


    def get_dev_examples(self, data_dir):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "dev", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "dev.json")), "dev", data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
        """See base class."""
        if self.args.pretrain:
            return self._create_examples(
                self._read_txt(os.path.join(self.args.pretrain_path, "wiki_tuple_ids.txt")), "test", data_dir, self.args)
        else:
            return self._create_examples(
                self._read_json(os.path.join(data_dir, "test.json")), "test", data_dir, self.args)


    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        return list(rel2token.values())
    
    def get_analogy_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip().split('\t')[0])
                
        with open(os.path.join(data_dir, "analogy_relations.txt"), 'r') as f:
            lines = f.readlines()
            analogy_relations = []
            for line in lines:
                analogy_relations.append(line.strip().replace('\n', ''))
                
        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations) if ent in analogy_relations}
        return list(rel2token.values())

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(self.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        return list(ent2token.values())
    
    def get_analogy_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(self.entity_path, 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip().split("\t")[0])
                
        with open(os.path.join(data_dir, "analogy_entities.txt"), 'r') as f:
            lines = f.readlines()
            analogy_entities = []
            for line in lines:
                analogy_entities.append(line.strip().replace('\n', ''))
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities) if ent in analogy_entities}
        return list(ent2token.values())
    
    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        relation = []
        with open(os.path.join(self.args.pretrain_path, "relation2text.txt"), 'r') as f:
            lines = f.readlines()
            for line in lines:
                relation.append(line.strip().split("\t")[-1])
        return relation

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = self._read_dict_txt(self.entity_path, quotechar='\t')
        # entities
        entities = list(ent2text.keys())
        # entity to virtual token
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        # entity to id
        ent2id = {ent : i for i, ent in enumerate(entities)}
        
        # relation to text
        rel2text = self._read_dict_txt(os.path.join(self.args.pretrain_path, "relation2text.txt"))
        
        # rel id -> relation token id
        rel2id = {rel: i for i, rel in enumerate(rel2text.keys())}

        # anlogy entities and relations
        with open(os.path.join(data_dir, "analogy_entities.txt"), 'r') as f:
            analogy_entities = []
            for line in f.readlines():
                analogy_entities.append(line.strip().replace('\n', ''))
        analogy_ent2id, i = {}, 0
        for ent in entities:
            if ent in analogy_entities:
                analogy_ent2id[ent] = i
                i += 1
        
        with open(os.path.join(data_dir, "analogy_relations.txt"), 'r') as f:
            analogy_relations = []
            for line in f.readlines():
                analogy_relations.append(line.strip().replace('\n', ''))
        analogy_rel2id, i = {}, 0
        for rel in rel2id:
            if rel in analogy_relations:
                rel2id[rel] = i
                i += 1

        examples = []
        filter_init(ent2text, rel2text, ent2id, ent2token, rel2id, analogy_ent2id, analogy_rel2id,
                    self.struct_entity2id, self.struct_relation2id, self.num_struct_entities)

        if args.pretrain:
            # delete entities without text name.
            tmp_lines = []
            not_in_text = 0
            for line in tqdm(lines, desc="delete entities without text name."):
                if (line[0] not in ent2text) or (line[2] not in ent2text) or (line[1] not in rel2text):
                    not_in_text += 1
                    continue
                tmp_lines.append(line)
            lines = tmp_lines
            print(f"total entity not in text : {not_in_text} ")

            examples = list(
                tqdm(
                    map(partial(solve, pretrain=self.args.pretrain), lines),
                    total=len(lines),
                    desc="Pretrain pre_type=1, convert text to examples"
                )
            )

        else:
            examples = list(
                tqdm(
                    map(partial(solve, pretrain=self.args.pretrain), lines),
                    total=len(lines),
                    desc="Fine-tuning, convert text to examples"
                )
            )

        # flatten examples
        examples = [sub_example for example in examples for sub_example in example]
        # delete vars
        del ent2text, rel2text, ent2id, ent2token, rel2id
        return examples


class KGCDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return len(self.features)


class MultiprocessingEncoder(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.pretrain = args.pretrain
        self.max_seq_length = args.max_seq_length

    def initializer(self):
        global bpe
        bpe = self.tokenizer

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        lines: [InputExamples]
        """
        enc_lines = []
        enc_lines.append(self.convert_examples_to_features(example=lines))
        return enc_lines

    def convert_examples_to_features(self, example):
        pretrain = self.pretrain
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""
        
        text_a = example.text_a
        text_b = example.text_b
        text_c = example.text_c
        
        if self.pretrain:
            input_text_a = bpe.sep_token.join([text_a, text_b, text_c])
            input_text_b = None
            inputs = bpe(
                input_text_a,
                input_text_b,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
            input_ids = inputs["input_ids"]
            # a. 获取实体对应的特殊token名称, e.g., '[ENTITY_123]'
            # ent2token 是一个全局变量，在 _create_examples 中被 filter_init 设置
            head_ent_token_name = ent2token.get(example.head_ent)
            tail_ent_token_name = ent2token.get(example.tail_ent)

            # b. 从tokenizer词汇表中获取该token的ID
            head_ent_token_id = bpe.vocab[head_ent_token_name] if head_ent_token_name and head_ent_token_name in bpe.vocab else -1
            tail_ent_token_id = bpe.vocab[tail_ent_token_name] if tail_ent_token_name and tail_ent_token_name in bpe.vocab else -1
            
            # c. 在input_ids中查找ID的位置
            # 使用 try-except 比 if-else 更简洁安全
            try:
                head_idx = input_ids.index(head_ent_token_id)
            except (ValueError, TypeError):
                head_idx = -1
            
            try:
                tail_idx = input_ids.index(tail_ent_token_id)
            except (ValueError, TypeError):
                tail_idx = -1
                
            features = asdict(InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                label=example.real_label,
                head_id=example.head_id,
                rel_id=example.rel_id,
                tail_id=example.tail_id,
                head_ent=example.head_ent,
                tail_ent=example.tail_ent,
                pre_type=example.pre_type,
                struct_head_id=example.struct_head_id,
                struct_rel_id=example.struct_rel_id,
                struct_tail_id=example.struct_tail_id,
                # +++ 传入新找到的位置 +++
                head_token_idx=head_idx,
                tail_token_idx=tail_idx
            ))
        else:
            text_d, text_e, text_f = example.text_d, example.text_e, example.text_f
            
            input_text_a = bpe.sep_token.join([text_a, text_b, text_c])
            input_text_b = bpe.sep_token.join([text_d, text_e, text_f])
        
            inputs = bpe(
                input_text_a,
                input_text_b,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
            # +++ 添加此代码块以查找token索引 +++
            input_ids = inputs["input_ids"]
            # 微调任务涉及一个范例(q_head, q_tail)和一个问题(a_head)
            # 我们需要这三个实体的特殊token名称
            # q_head/q_tail的实体名存储在example.head_ent和example.tail_ent中
            q_head_token_name = ent2token.get(example.head_ent) if example.head_ent in ent2token else None
            q_tail_token_name = ent2token.get(example.tail_ent) if example.tail_ent in ent2token else None
            
            # a_head的实体名需要通过ID反查
            a_head_ent_name = None
            for name, idx in ent2id.items():
                if idx == example.a_head_id:
                    a_head_ent_name = name
                    break
            a_head_token_name = ent2token.get(a_head_ent_name) if a_head_ent_name in ent2token else None

            # 从词汇表中获取token ID
            q_head_token_id = bpe.vocab.get(q_head_token_name, -1)
            q_tail_token_id = bpe.vocab.get(q_tail_token_name, -1)
            a_head_token_id = bpe.vocab.get(a_head_token_name, -1)
            
            # 在token序列中查找索引
            try:
                q_head_idx = input_ids.index(q_head_token_id) if q_head_token_id != -1 else -1
            except ValueError:
                q_head_idx = -1
            try:
                q_tail_idx = input_ids.index(q_tail_token_id) if q_tail_token_id != -1 else -1
            except ValueError:
                q_tail_idx = -1
            try:
                a_head_idx = input_ids.index(a_head_token_id) if a_head_token_id != -1 else -1
            except ValueError:
                a_head_idx = -1
            
            features = asdict(AnalogyInputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs['token_type_ids'],
                label=example.real_label,
                rel_label=example.relation,
                q_head_id=example.q_head_id,
                q_tail_id=example.q_tail_id,
                a_head_id=example.a_head_id,
                head_ent=example.head_ent,
                tail_ent=example.tail_ent,
                # +++ 传入所有新的结构化ID和token索引 +++
                struct_q_head_id=example.struct_q_head_id,
                struct_q_tail_id=example.struct_q_tail_id,
                struct_a_head_id=example.struct_a_head_id,
                struct_rel_id=example.struct_rel_id,
                q_head_token_idx=q_head_idx,
                q_tail_token_idx=q_tail_idx,
                a_head_token_idx=a_head_idx
            ))
        assert bpe.mask_token_id in inputs.input_ids, "mask token must in input"

        return features
