import torch
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from spert import sampling
from spert import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) 这是一个辅助函数，用于从BERT模型的隐藏状态中提取特定token（例如[CLS]）的嵌入表示。 """
    emb_size = h.shape[-1]#隐藏状态的最后一个维度大小，也就是嵌入维度

    token_h = h.view(-1, emb_size)#将隐藏状态张量展平为二维张量，形状为(total_tokens, emb_size)
    flat = x.contiguous().view(-1)#[N, context_size]-->[N*context_size,]

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]#从展平的输入张量中找到所有等于指定token的索引，并从展平的隐藏状态张量中提取对应的嵌入表示，一般取出的是CLS的表示，因为bert模型的第一个token是CLS，输出的特征可以表示全文的语义信息

    return token_h


class SpERT(BertPreTrainedModel):
    """ Span-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SpERT, self).__init__(config)

        # BERT model创建基础的语言模型，将输入的token ids和attention masks传入BERT模型，获取上下文相关的token嵌入表示。
        self.bert = BertModel(config)

        # layers
        # [BERT输出的隐藏状态维度的3倍加上大小嵌入维度的2倍，映射到关系类型的数量]这里就是一个实体，一个宽度编码，一个全局上下文
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        #实体分类器：将实体候选表示映射到实体类型的数量
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        #实体大小嵌入：将实体的大小（长度）映射到一个固定维度的嵌入表示
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token#cls_token的id
        self._relation_types = relation_types#关系类型数量
        self._entity_types = entity_types#实体类型数量
        self._max_pairs = max_pairs#每次处理的最大关系对数量

        # weight initialization 参数初始化
        self.init_weights()

        if freeze_transformer:#是否冻结BERT模型的参数，不进行更新
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        '''
        encodings:原始的token编码，形状为(context_size,)
        context_masks:上下文掩码，形状为(context_size,)
        entity_masks:实体掩码，表示对每一个实体进行掩码，包括正样本和负样本，形状为(num_entities, context_size)，如果某个位置属于实体范围内则为1，否则为0，并且负样本实体的掩码全部为0
        entity_sizes:实体大小，表示总共有多少个实体且每个样本的长度，形状为(num_entities,)

        rels:关系对索引，形状为(num_relations, 2)，包括正样本和负样本，每个关系对由两个实体的索引组成
        rel_masks:关系掩码，形状为(num_relations, context_size)，每一个关系对应一个掩码，表示关系对应的文本区域进行掩码（两个实体之间的文本区域之间的所以位置为1）
        '''
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']#这里获取BERT模型的最后一层隐藏状态

        batch_size = encodings.shape[0]

        # classify entities
        #输入特征是实体掩码和实体大小嵌入，输出是实体分类的logits和实体候选的池化表示
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes 这里是将实体的大小进行嵌入表示
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)#得到实体分类的logits和实体候选的池化表示

        # classify relations
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']#获取BERT模型的最后一层隐藏状态

        batch_size = encodings.shape[0]#获取批次大小
        ctx_size = context_masks.shape[-1]#获取上下文的长度

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes) # embed entity candidate sizes 这里是将实体的大小进行嵌入表示
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)#得到实体分类的logits和实体候选的池化表示

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)  这一步是根据实体分类的结果，过滤掉那些不被分类为实体的候选实体，只保留被分类为实体的候选实体，用于后续的关系分类
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage 因为关系对的数量可能非常大，所以这里采用分块处理的方式，逐块对关系对进行分类
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans 这里需要将[N, num_entities, context_size]的实体掩码转换为[N, num_entities, context_size, 1]，然后与h[N, context_size, hidden_size]进行广播相加，最后在context_size维度上进行max pooling，得到实体候选的池化表示
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)#将实体掩码中为0的位置设置为一个很小的值，为1的地方设置为0[N, num_entities, context_size]-->[N, num_entities, context_size, 1]
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)#h[N, context_size, hidden_size]-->[N, num_entities, context_size, hidden_size],这里表示n个批次，每一个实体的特征都是T*128的特征，然后与实体掩码进行广播相加
        #相加后的结果是对bert的实体位置进行保留，非实体位置被设置为一个很小的值
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]#针对每一个实体的context_size维度，也就是时间维度进行max pooling，得到实体候选的池化表示[N, num_entities, hidden_size],

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)#从BERT的隐藏状态中提取CLS token的嵌入表示，作为实体候选的上下文表示[N, hidden_size]

        # create candidate representations including context, max pooled span and size embedding ，这里将实体候选的基于cls提取的上下文特征、实体范围的池化表示特征和实体大小嵌入特征进行拼接，得到最终的实体候选表示
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates 通过全连接得到每一个实体属于各个实体类型的置信度
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0] #获取批次大小

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs: #如果关系对的数量超过了最大处理数量，就进行分块处理
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations  这里相当于论文里面的红色特征部分 获取对应的实体对的表示
        entity_pairs = util.batch_index(entity_spans, relations)#[N,num_entity_pairs, hidden_size] 和[N, num_entity_pairs, 2],这里是根据关系对的实体索引，从实体候选表示中提取对应的实体表示特征，变成[N, num_entity_pairs, 2, hidden_size]
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)#将实体对的表示进行展平，变成[N, num_entity_pairs, hidden_size*2]

        # get corresponding size embeddings 这里相当于论文里面的蓝色特征部分，获取对应的实体对的大小嵌入表示
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens  这里相当于论文里面的黄色特征部分，获取实体对之间的上下文表示 这里的方法和刚才的提取实体候选的池化表示类似
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero,需要注意对于负样本来说，实体对之间没有上下文，所以需要将这些位置的上下文表示设置为0
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none) 这里是获取每一个实体候选的最大置信度的实体类型索引，并且乘以实体样本掩码，过滤掉那些不被分类为实体的候选实体
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)#获取被分类为实体的候选实体的索引，这里假设实体类型索引0表示非实体，所以非零索引表示被分类为实体的候选实体
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()#获取被分类为实体的候选实体的跨度信息
            non_zero_indices = non_zero_indices.tolist()#将张量转换为列表

            # create relations and masks 将被分类为实体的候选实体两两组合，形成关系对，并且为每一个关系对创建对应的关系掩码
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack 填充批次中的关系对、关系掩码和关系样本掩码，使得它们具有相同的长度
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'spert': SpERT,
}


def get_model(name):
    return _MODELS[name]

def load_model(self,tokenizer, input_reader):
    model_class = get_model(self.model_type)#获取模型类

    config = BertConfig.from_pretrained(self.model_path, cache_dir=self.cache_path)#加载预训练模型的配置文件
    util.check_version(config, model_class, self.model_path)#检查版本

    config.spert_version = model_class.VERSION
    model = model_class.from_pretrained(self.model_path,
                                        config=config,
                                        # SpERT model parameters
                                        cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),
                                        relation_types=input_reader.relation_type_count - 1,
                                        entity_types=input_reader.entity_type_count,
                                        max_pairs=self.max_pairs,
                                        prop_drop=self.prop_drop,
                                        size_embedding=self.size_embedding,
                                        freeze_transformer=self.freeze_transformer,
                                        cache_dir=self.cache_path)#加载预训练模型，模型恢复

    return model
