import random

import torch

from spert import util


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
    encodings = doc.encoding#获取文本编码
    token_count = len(doc.tokens)#获取文本中token的数量
    context_size = len(encodings)#获取编码的长度

    # positive entities 正样本的构建
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)#获取实体的跨度，也就是起始和结束位置
        pos_entity_types.append(e.entity_type.index)#获取实体的类型索引
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))#创建实体掩码
        pos_entity_sizes.append(len(e.tokens))#获取实体的大小

    # positive relations

    # collect relations between entity pairs 创建实体对之间的关系正样本，但是这里的建立是单关系的实体对
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)#获取关系的头实体和尾实体组成的实体对
        if pair not in entity_pair_relations:
             entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples 构建正样本的关系
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair #获取实体对的头实体和尾实体
        s1, s2 = head_entity.span, tail_entity.span #获取头实体和尾实体的跨度
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2))) #将实体跨度转换为索引并添加到正样本关系列表中
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]#获取实体对之间的关系类型索引，这里可能有多个关系类型
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]#将关系类型转换为多标签二进制表示，忽略类型0
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))#创建关系掩码，对关系对应的文本区域进行掩码（两个实体之间的文本区域之间的所以位置为1）

    # negative entities 负样本实体的构建
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_entity_spans:#确保负样本实体不在正样本实体中
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))#选择指定数量的负样本实体，最小值为负样本实体总数和所需负样本数量
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # negative relations  关系负样本的构建
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))
            #当实体对不是同一个实体且不在正样本关系跨度中时，添加为负样本关系跨度，选择正样本实体对之间不存在关系的实体对作为负样本关系，也就是强负样本关系，不要那一个正确的样本的其余组合
    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))#选择指定数量的负样本关系，最小值为负样本关系总数和所需负样本数量

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count-1)] * len(neg_rel_spans)

    # merge 正负样本的合并
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)
    '''
    encodings:原始的token编码，形状为(context_size,)
    context_masks:上下文掩码，形状为(context_size,)
    entity_masks:实体掩码，表示对每一个实体进行掩码，包括正样本和负样本，形状为(num_entities, context_size)，如果某个位置属于实体范围内则为1，否则为0，并且负样本实体的掩码全部为0
    entity_sizes:实体大小，表示总共有多少个实体且每个样本的长度，形状为(num_entities,)
    entity_types:实体类型索引，形状为(num_entities,)
    entity_sample_masks:实体样本掩码，形状为(num_entities,)
    
    rels:关系对索引，形状为(num_relations, 2)，包括正样本和负样本，每个关系对由两个实体的索引组成
    rel_masks:关系掩码，形状为(num_relations, context_size)，每一个关系对应一个掩码，表示关系对应的文本区域进行掩码（两个实体之间的文本区域之间的所以位置为1）
    rel_types:关系类型的多标签二进制表示，形状为(num_relations, rel_type_count-1)，eg：[0,1,0,1]表示该关系同时属于类型1和类型3,所以负样本的关系类型全部为0
    rel_sample_masks:关系样本掩码，形状为(num_relations,)
    '''

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()#提取字典的键

    for key in keys:
        samples = [s[key] for s in batch]#提取每个样本中对应键的值
        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])#对样本进行填充堆叠，包括多维数据

    return padded_batch
