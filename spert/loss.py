from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion#关系损失函数，是二元交叉熵损失函数
        self._entity_criterion = entity_criterion#实体损失函数，是交叉熵损失函数
        self._model = model#模型
        self._optimizer = optimizer#优化器
        self._scheduler = scheduler#学习率调度器
        self._max_grad_norm = max_grad_norm#最大梯度范数

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss 计算实体识别损失
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])#将实体logits展平为二维张量，形状为(all_num_entities, num_entity_types)
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()#将实体损失按样本掩码加权平均，对于不是实体的样本不计算损失

        # relation loss 计算关系分类损失
        rel_sample_masks = rel_sample_masks.view(-1).float()#将关系样本掩码展平为一维张量
        rel_count = rel_sample_masks.sum()#计算有效的关系样本数量

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])#将关系logits展平为二维张量，形状为(all_num_relations, num_relation_types)
            rel_types = rel_types.view(-1, rel_types.shape[-1])#将关系类型展平为二维张量，形状为(all_num_relations, num_relation_types)

            rel_loss = self._rel_criterion(rel_logits, rel_types)#计算关系损失，形状为(all_num_relations, num_relation_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]#对多标签损失进行平均，得到每个关系的平均损失，形状为(all_num_relations,)
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count#将关系损失按样本掩码加权平均，对于不是关系的样本不计算损失

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()
