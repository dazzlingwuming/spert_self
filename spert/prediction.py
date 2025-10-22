import json
from typing import Tuple

import torch

from spert import util
from spert.input_reader import BaseInputReader


def convert_predictions(batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                        batch_rels: torch.tensor, batch: dict, rel_filter_threshold: float,
                        input_reader: BaseInputReader, no_overlapping: bool = False):
    # get maximum activation (index of predicted entity type)
    batch_entity_types = batch_entity_clf.argmax(dim=-1)#选择每个实体的最大激活值对应的索引作为预测的实体类型
    # apply entity sample mask
    batch_entity_types *= batch['entity_sample_masks'].long()

    # apply threshold to relations
    batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0#选择关系分类器输出中小于阈值的部分设为0，表示这些关系不被预测为有效关系

    batch_pred_entities = []
    batch_pred_relations = []

    for i in range(batch_rel_clf.shape[0]):
        # get model predictions for sample
        entity_types = batch_entity_types[i]#获取每个样本的实体类型预测
        entity_spans = batch['entity_spans'][i]#获取每个样本的实体跨度
        entity_clf = batch_entity_clf[i]#获取每个样本的实体分类器输出
        rel_clf = batch_rel_clf[i]#获取每个样本的关系分类器输出
        rels = batch_rels[i]#获取每个样本的关系对索引

        # convert predicted entities 提取实体，输出是一个包含实体信息的列表，每个实体信息包括起始位置、结束位置、实体类型和置信度分数
        sample_pred_entities = _convert_pred_entities(entity_types, entity_spans,
                                                      entity_clf, input_reader)

        # convert predicted relations 提取关系，输出是一个包含关系信息的列表，每个关系信息包括头实体信息、尾实体信息、关系类型和置信度分数
        sample_pred_relations = _convert_pred_relations(rel_clf, rels,
                                                        entity_types, entity_spans, input_reader)

        if no_overlapping:
            sample_pred_entities, sample_pred_relations = remove_overlapping(sample_pred_entities,
                                                                             sample_pred_relations)

        batch_pred_entities.append(sample_pred_entities)
        batch_pred_relations.append(sample_pred_relations)

    return batch_pred_entities, batch_pred_relations


def _convert_pred_entities(entity_types: torch.tensor, entity_spans: torch.tensor,
                           entity_scores: torch.tensor, input_reader: BaseInputReader):
    # get entities that are not classified as 'None'
    valid_entity_indices = entity_types.nonzero().view(-1)
    pred_entity_types = entity_types[valid_entity_indices]
    pred_entity_spans = entity_spans[valid_entity_indices]
    pred_entity_scores = torch.gather(entity_scores[valid_entity_indices], 1,
                                      pred_entity_types.unsqueeze(1)).view(-1)

    # convert to tuples (start, end, type, score)
    converted_preds = []
    for i in range(pred_entity_types.shape[0]):
        label_idx = pred_entity_types[i].item()
        entity_type = input_reader.get_entity_type(label_idx)

        start, end = pred_entity_spans[i].tolist()
        score = pred_entity_scores[i].item()

        converted_pred = (start, end, entity_type, score)
        converted_preds.append(converted_pred)

    return converted_preds


def _convert_pred_relations(rel_clf: torch.tensor, rels: torch.tensor,
                            entity_types: torch.tensor, entity_spans: torch.tensor, input_reader: BaseInputReader):
    rel_class_count = rel_clf.shape[1]#关系类别的数量
    rel_clf = rel_clf.view(-1)#将关系分类器的输出张量展平为一维张量，以便后续处理 [num_relations * rel_class_count]

    # get predicted relation labels and corresponding entity pairs
    rel_nonzero = rel_clf.nonzero().view(-1)#获取所有非零的关系预测索引，这些索引对应于被预测为有效关系的实体对
    pred_rel_scores = rel_clf[rel_nonzero]

    pred_rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1) 这里获取所有非零位置下的预测类别
    valid_rel_indices = rel_nonzero // rel_class_count #获取关系实体实际id
    valid_rels = rels[valid_rel_indices] #提取最终预测的实体对

    # get masks of entities in relation
    pred_rel_entity_spans = entity_spans[valid_rels].long()#提取两个实体的索引范围

    # get predicted entity types 对应实体的类型，提取的是实体对中每个实体的类型（预测）
    pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
    if valid_rels.shape[0] != 0:
        pred_rel_entity_types = torch.stack([entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])])

    # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
    converted_rels = []
    check = set()

    for i in range(pred_rel_types.shape[0]):
        label_idx = pred_rel_types[i].item()#获取关系类型的预测索引
        pred_rel_type = input_reader.get_relation_type(label_idx)#根据索引获取关系类型对象
        pred_head_type_idx, pred_tail_type_idx = pred_rel_entity_types[i][0].item(), pred_rel_entity_types[i][1].item()#获取头实体和尾实体的类型索引
        pred_head_type = input_reader.get_entity_type(pred_head_type_idx)#获取头实体的类型对象
        pred_tail_type = input_reader.get_entity_type(pred_tail_type_idx)#获取尾实体的类型对象
        score = pred_rel_scores[i].item()#获取关系的置信度分数

        spans = pred_rel_entity_spans[i]
        head_start, head_end = spans[0].tolist()
        tail_start, tail_end = spans[1].tolist()

        converted_rel = ((head_start, head_end, pred_head_type),
                         (tail_start, tail_end, pred_tail_type), pred_rel_type)
        converted_rel = _adjust_rel(converted_rel)

        if converted_rel not in check:
            check.add(converted_rel)
            converted_rels.append(tuple(list(converted_rel) + [score]))

    return converted_rels


def remove_overlapping(entities, relations):
    non_overlapping_entities = []
    non_overlapping_relations = []

    for entity in entities:
        if not _is_overlapping(entity, entities):
            non_overlapping_entities.append(entity)

    for rel in relations:
        e1, e2 = rel[0], rel[1]
        if not _check_overlap(e1, e2):
            non_overlapping_relations.append(rel)

    return non_overlapping_entities, non_overlapping_relations


def _is_overlapping(e1, entities):
    for e2 in entities:
        if _check_overlap(e1, e2):
            return True

    return False


def _check_overlap(e1, e2):
    if e1 == e2 or e1[1] <= e2[0] or e2[1] <= e1[0]:
        return False
    else:
        return True


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel
    if rel[-1].symmetric:
        head, tail = rel[:2]
        if tail[0] < head[0]:
            adjusted_rel = tail, head, rel[-1]

    return adjusted_rel


def store_predictions(documents, pred_entities, pred_relations, store_path):
    predictions = text_predictions(documents, pred_entities, pred_relations)

    # store as json
    with open(store_path, 'w') as predictions_file:
        json.dump(predictions, predictions_file)

def text_predictions(documents, pred_entities, pred_relations):
    predictions = []

    for i, doc in enumerate(documents):
        tokens = doc.tokens
        sample_pred_entities = pred_entities[i]
        sample_pred_relations = pred_relations[i]

        # convert entities
        converted_entities = []
        for entity in sample_pred_entities:
            entity_span = entity[:2]
            span_tokens = util.get_span_tokens(tokens, entity_span)
            entity_type = entity[2].identifier
            converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_entities.append(converted_entity)
        converted_entities = sorted(converted_entities, key=lambda e: e['start'])

        # convert relations
        converted_relations = []
        for relation in sample_pred_relations:
            head, tail = relation[:2]
            head_span, head_type = head[:2], head[2].identifier
            tail_span, tail_type = tail[:2], tail[2].identifier
            head_span_tokens = util.get_span_tokens(tokens, head_span)
            tail_span_tokens = util.get_span_tokens(tokens, tail_span)
            relation_type = relation[2].identifier

            converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                  end=head_span_tokens[-1].index + 1)
            converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                  end=tail_span_tokens[-1].index + 1)

            head_idx = converted_entities.index(converted_head)
            tail_idx = converted_entities.index(converted_tail)

            converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
            converted_relations.append(converted_relation)
        converted_relations = sorted(converted_relations, key=lambda r: r['head'])

        doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=converted_entities,
                               relations=converted_relations)
        predictions.append(doc_predictions)

    return predictions
