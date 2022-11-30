"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
import random
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
import torch
from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
from utils.convert_sample2ids import sample2ids
from utils.logger import LOGGER


class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """

    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        # self.uniter1 = UniterModel(config, img_dim)
        self.transformerLayer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.w = nn.Parameter(torch.ones(2))  # 用于对[cls]和[平均表征]加权求和
        self.sampleIds = sample2ids()  # 根据分类结果填充对应Topic/Sample id
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ###########################
        self.vqa_norm = LayerNorm(config.hidden_size * 2, eps=1e-12)
        self.vqa_linear = nn.Linear(config.hidden_size * 2, num_answer)
        ###########################
        self.itm_norm = LayerNorm(256, eps=1e-12)
        self.itm_linear = nn.Linear(256, 25)
        ###########################
        self.prompt_norm = LayerNorm(256, eps=1e-12)
        self.prompt_linear = nn.Linear(256, 25)
        ###########################
        with open("vocab.txt", "r") as f:
            self.vocab = f.read().splitlines()
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            GELU(),
            # LayerNorm(config.hidden_size * 2, eps=1e-12),
            # nn.Linear(config.hidden_size * 2, num_answer)
        )
        self.mask_pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        # 将CLS输出映射到3129个备选答案上
        self.itm_output = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            GELU(),
            # LayerNorm(256, eps=1e-12),
            # nn.Linear(256, 25)
        )
        self.prompt_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            GELU(),
            # LayerNorm(256, eps=1e-12),
            # nn.Linear(256, 25)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        # (batchsize, xxx)
        input_ids = batch['input_ids']  # 问题 + 弱模版
        # 因为在经过topic分类器后需要修改input_ids，因此必须创建一个新的input_ids
        # 避免使用inplace-operation操作修改需要参与梯度计算的变量
        input_ids_topic = input_ids.detach().clone()
        # 构造负样例
        neg_input_ids_list = [input_ids.detach().clone()]
        # neg_input_ids = input_ids.detach().clone()
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        template_ids = batch["template_ids"]  # 是模版分类的标签
        mask_pos = batch['mask_pos']  # [MASK]索引
        targets = batch['targets']

        # 先做分类，得到分类结果——第几种模版
        # 输入就是img feat + txt feat
        # 调用uniter计算特征的函数直接得到特征送进分类器
        # 根据分类结果将Topic和答案示例id填入[MASK]
        # sequence_output, attention_probs = self.uniter1(input_ids, position_ids,
        #                                                img_feat, img_pos_feat,
        #                                                attn_masks, template_ids=None,
        #                                                output_all_encoded_layers=False)
        # topic_ids = [mask[1] for mask in mask_pos]  # [topic]索引
        # mask_tokens = torch.stack([sequence_output[i][topic_ids[i]] for i in range(len(topic_ids))])
        # pooled_output = self.mask_pooler(mask_tokens)
        # cls_output = self.uniter.pooler(sequence_output)
        # # prompt_kind: [x ,x ,x ...],长度：batch_size
        # topic_output = self.prompt_classifier(pooled_output)
        # prompt_kind = topic_output.max(1)[1]

        """使用transformerLayer做分类"""
        embedding_output, img_len = self.uniter.compute_img_txt_embeddings(input_ids_topic, position_ids,
                                                                            img_feat, img_pos_feat, gather_index=None, )
        # print("embedding_output:" + str(embedding_output.shape)) （768维）
        embedding_output = torch.transpose(embedding_output, 0, 1)
        out = self.transformerLayer(embedding_output)
        out = torch.transpose(out, 0, 1)
        cls = out[:, 0, :]
        out = out.mean(1)
        # 归一化权重
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        out = w1 * cls + w2 * out
        out = self.prompt_classifier(out)
        topic_output = self.prompt_linear(self.prompt_norm(out.float()))
        prompt_kind = topic_output.max(1)[1]

        # prompt_kind: [x ,x ,x ...],长度：batch_size
        # mask_pos是一个List, 每一个元素是一个长度为5的tensor
        # 根据分类结果，将对应的topic及sample id填入input_id中的[MASK]位置
        # print("prompt_kind:" + str(prompt_kind))

        # 构造正样例
        answer_ids = []
        for i, input_id in enumerate(input_ids):
            pos = mask_pos[i]  # 长度为5的Tensor
            # sample_ids = self.sampleIds[prompt_kind[i]]  # 长度为4的List
            input_id[pos[1]] = self.sampleIds.get_topic(prompt_kind[i])
            sample_ids = self.sampleIds.get_sample(prompt_kind[i])
            input_id[pos[2]], input_id[pos[3]], input_id[pos[4]] = sample_ids[0], sample_ids[1], sample_ids[2]
            answer_ids.append(pos[0])
            # 输出验证
            # text = " ".join([self.vocab[id] for id in input_id])
            # print("input:" + text)

        """
        # 构造负样例
        offset = random.randint(1, 24)
        for i, neg_input_id in enumerate(neg_input_ids):
            pos = mask_pos[i]
            neg_input_id[pos[1]] = self.sampleIds.get_topic((prompt_kind[i] + offset) % 24)
            sample_ids = self.sampleIds.get_sample((prompt_kind[i] + offset) % 24)
            neg_input_id[pos[2]], neg_input_id[pos[3]], neg_input_id[pos[4]] = sample_ids[0], sample_ids[1], \
                                                                                   sample_ids[2]
        """

        # 训练正样例
        sequence_output, attention_probs = self.uniter(input_ids, position_ids,
                                                       img_feat, img_pos_feat,
                                                       attn_masks,
                                                       output_all_encoded_layers=False,
                                                       template_ids=prompt_kind)
        # 将[MASK]映射到答案数量维度
        mask_tokens = torch.stack([sequence_output[i][answer_ids[i]] for i in range(len(answer_ids))])
        pooled_output = self.mask_pooler(mask_tokens)
        # answer_scores = self.vqa_output(pooled_output)
        answer_scores = self.vqa_linear(self.vqa_norm(self.vqa_output(pooled_output).float()))
        # 将[CLS]映射到 yes/no
        cls_output = self.uniter.pooler(sequence_output)
        # itm_scores = self.itm_output(cls_output)
        itm_scores = self.itm_linear(self.itm_norm(self.itm_output(cls_output).float()))

        """
        # 训练负样例
        sequence_output, attention_probs = self.uniter(neg_input_ids, position_ids,
                                                        img_feat, img_pos_feat,
                                                        attn_masks,
                                                        output_all_encoded_layers=False,
                                                        template_ids=prompt_kind)

        # Neg MLM loss
        neg_mask_tokens = torch.stack([sequence_output[i][answer_ids[i]] for i in range(len(answer_ids))])
        neg_answer_scores = self.vqa_output(self.mask_pooler(neg_mask_tokens))
        neg_itm_scores = self.itm_output(self.uniter.pooler(sequence_output))


        neg_vqa_loss = F.binary_cross_entropy_with_logits(
                neg_answer_scores, targets, reduction='none')
        # Neg ITM loss
        one_hots = torch.zeros(*targets.size(), device=targets.device)
        one_hots.scatter_(1, torch.max(neg_answer_scores, 1)[1].view(-1, 1), 1)
        neg_itm_label = (one_hots * targets).sum(dim=1).long()
        neg_itm_loss = F.cross_entropy(neg_itm_scores, neg_itm_label)

        neg_loss = neg_vqa_loss.mean() + neg_itm_loss
        """

        # 构造并训练负样例
        neg_loss = 0
        # for neg_input_ids in neg_input_ids_list:
        #     offset = random.randint(1, 24)
        #     for i, neg_input_id in enumerate(neg_input_ids):
        #         pos = mask_pos[i]
        #         neg_input_id[pos[1]] = self.sampleIds.get_topic((prompt_kind[i] + offset) % 24)
        #         sample_ids = self.sampleIds.get_sample((prompt_kind[i] + offset) % 24)
        #         neg_input_id[pos[2]], neg_input_id[pos[3]], neg_input_id[pos[4]] = sample_ids[0], sample_ids[1], \
        #                                                                            sample_ids[2]
        #     sequence_output, attention_probs = self.uniter(neg_input_ids, position_ids,
        #                                                    img_feat, img_pos_feat,
        #                                                    attn_masks,
        #                                                    output_all_encoded_layers=False,
        #                                                    template_ids=prompt_kind)
        #     neg_mask_tokens = torch.stack([sequence_output[i][answer_ids[i]] for i in range(len(answer_ids))])
        #     neg_answer_scores = self.vqa_output(self.mask_pooler(neg_mask_tokens))
        #     neg_answer_scores = self.vqa_linear(self.vqa_norm(neg_answer_scores.float()))
        #     #####################################################################
        #     neg_itm_scores = self.itm_output(self.uniter.pooler(sequence_output))
        #     neg_itm_scores = self.itm_linear(self.itm_norm(neg_itm_scores.float()))
        #
        #
        #     # Neg MLM loss: sigmoid -> 点积 -> BCE loss
        #     # neg_answer_scores = self.sigmoid(neg_answer_scores)
        #     # neg_answer_scores = neg_answer_scores * targets
        #     # # loss计算
        #     # neg_vqa_loss = F.binary_cross_entropy(
        #     #     neg_answer_scores, targets, reduction='none')
        #     # Neg ITM loss: get labels -> score softmax -> expand labels -> 点积 -> NLL loss
        #     one_hots = torch.zeros(*targets.size(), device=targets.device)
        #     one_hots.scatter_(1, torch.max(neg_answer_scores, 1)[1].view(-1, 1), 1)
        #     neg_itm_label = (one_hots * targets).sum(dim=1).long()
        #     neg_itm_scores = self.softmax(neg_itm_scores)  # softmax
        #     expand_labels = neg_itm_label.view(-1, 1).expand_as(neg_itm_scores)  # expand labels
        #     neg_itm_scores = neg_itm_scores * expand_labels  # 点积
        #     # loss计算
        #     neg_itm_loss = F.nll_loss(neg_itm_scores, neg_itm_label)  # NLL loss
        #     # total loss
        #     # neg_loss = neg_loss + neg_vqa_loss.mean() + neg_itm_loss
        #     neg_loss = neg_loss + neg_itm_loss


        # 多标签分类问题：sigmoid + BCE loss
        if compute_loss:

            # MLM loss
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')

            # ITM loss
            one_hots = torch.zeros(*targets.size(), device=targets.device)
            one_hots.scatter_(1, torch.max(answer_scores, 1)[1].view(-1, 1), 1)
            itm_label = (one_hots * targets).sum(dim=1).long()
            itm_loss = F.cross_entropy(itm_scores, itm_label)

            # topic_loss
            template_ids = torch.tensor(template_ids, dtype=torch.long).cuda()
            topic_loss = F.cross_entropy(
                topic_output, template_ids, reduction='none'
            )

            return vqa_loss, itm_loss, topic_loss, neg_loss, attention_probs
        else:
            return answer_scores, attention_probs
