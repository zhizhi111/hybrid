"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
from re import template
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from utils.template import WTemplate
from pytorch_pretrained_bert import BertTokenizer

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index


def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


class VqaDataset(DetectFeatTxtTokDataset):
    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = num_answers
        # template - 模版类
        self.template = WTemplate()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
        with open("vocab.txt", "r") as f:
            self.vocab = f.read().splitlines()

        # --创建weak_prompt, 转化为id, 方便后续拼接-- #
        # [MASK]不能紧邻符号',' 否则分词器会将[MASK]分开
        weak_prompt = " , which is a [MASK] such as [MASK] [MASK] [MASK] , and we can notice that "
        self.weak_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(weak_prompt))


    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # input_ids: 陈述句
        input_ids = example['input_ids']
        # text：问题文本
        text = example['question']
        # print(example['question'])
        # 返回对应的input_ids 以及 拼接的弱模版
        # 返回template id和弱模版，其中template ID作为分类器监督标签

        template_id = self.template.weak_prompt(text)

        # 拼接上prompt_ids序列,得到输入：[CLS] 问题 + 弱提示 [SEP]
        input_ids = input_ids + self.weak_prompt_ids
        input_ids = self.txt_db.combine_inputs(input_ids)

        # 拿到[topic]和such as后[MASK]的位置
        mask_pos = torch.nonzero(input_ids == 103).squeeze()

        # 构造输入的attn_masks
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        target = _get_vqa_target(example, self.num_answers)

        return input_ids, img_feat, img_pos_feat, attn_masks, target, template_id, mask_pos


def vqa_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets, template_ids, mask_pos
     ) = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    #
    # print("input_ids:" + str(input_ids.shape))
    # print("position_ids:" + str(position_ids.shape))

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'img_feat': img_feat,
        'img_pos_feat': img_pos_feat,
        'attn_masks': attn_masks,
        'targets': targets,
        "template_ids": template_ids,
        "mask_pos": mask_pos
    }
    return batch


class VqaEvalDataset(VqaDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        # text：问题文本
        text = example['question']

        template_id = self.template.weak_prompt(text)

        # 拼接上prompt_ids序列,得到输入：[CLS] 问题 + 弱提示 [SEP]
        input_ids = input_ids + self.weak_prompt_ids
        input_ids = self.txt_db.combine_inputs(input_ids)

        # 拿到[topic]和such as后[MASK]的位置
        mask_pos = torch.nonzero(input_ids == 103).squeeze()

        # 构造attn_masks
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        if 'target' in example:
            target = _get_vqa_target(example, self.num_answers)
        else:
            target = None

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target, template_id, mask_pos


def vqa_eval_collate(inputs):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets, template_ids, mask_pos
     ) = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'targets': targets,
             "template_ids":template_ids,
             'mask_pos': mask_pos
             }
    return batch
