'''
Author: snowy
Email: 1447048804@qq.com
Date: 2025-05-27 11:56:38
Version: 0.0.1
LastEditors: snowy 1447048804@qq.com
LastEditTime: 2025-07-21 17:22:33
Description: 根据官方的实现，添加中文注释，源码：https://github.com/fundamentalvision/Deformable-DETR.git
'''

"""
Deformable-DETR 的 transformer 模块

从 torch.nn.Tramsformer 复制并粘贴过来后修改
    * 在 MHattention 中插入位置编码
    * 移除编码器末端的额外 LN 层
    * 解码器返回来自所有解码层的激活堆栈
"""
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False, num_feature_levels=4, dec_n_points=4,
                 enc_n_points=4, two_stage=False, two_stage_num_proposals=300):
        """
        d_model: 输入序列的维度
        nhead: 多头注意力的头数
        num_encoder_layers: 编码器层数
        num_decoder_layers: 解码器层数
        dim_feedforward: 前馈网络的隐藏层维度
        dropout: dropout 率
        activation: 激活函数
        return_intermediate_dec: 是否返回所有解码器层的中间输出（用于深度监督）
        num_feature_levels: 特征图的层数
        dec_n_points: 解码器每层的点数, 相当于采样的 K 数
        enc_n_points: 编码器每层的点数, 相当于采样的 K 数
        two_stage: 是否使用两阶段策略
        two_stage_num_proposals: 两阶段策略的提议数
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model)) # 层的位置编码

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        '''计算掩码的有效区域比例， 防止图像因为padding而导致的掩码不准确或无效'''
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) # 高度方向有效像素
        valid_W = torch.sum(~mask[:, 0, :], 1) # 宽度方向有效像素
        valid_ratio_h = valid_H.float() / H # 高度方向有效像素比例
        valid_ratio_w = valid_W.float() / W # 宽度方向有效像素比例
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # [bs, 2]
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        srcs: [[bs, 256, h/8, w/8], [bs, 256, h/16, w/16], [bs, 256, h/32, w/32], [bs, 256, h/64, w/64]]
        masks: [[bs, h/8, w/8], [bs, h/16, w/16], [bs, h/32, w/32], [bs, h/64, w/64]]
        pos_embeds: [[bs, 256, h/8, w/8], [bs, 256, h/16, w/16], [bs, 256, h/32, w/32], [bs, 256, h/64, w/64]]
        query_embed: [300, 512]
        """
        assert self.two_stage or query_embed is not None

        # 准备 encoder 的输入
        src_flatten = [] # 多层特征图展开后合并的 token 序列
        mask_flatten = [] # 多层特征图对应的 mask 展开后合并
        lvl_pos_embed_flatten = [] # 多层特征图对应的位置编码展开后合并（层位置编码和特征图位置编码相加）
        spatial_shapes = [] # 多层特征图的尺寸
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # [bs, c, h, w] -> [bs, h*w, c]
            mask = mask.flatten(1) # [bs, h, w] -> [bs, h*w]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [bs, c, h, w] -> [bs, h*w, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # [bs, h*w, c]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1) # [bs, sum(h*w across levels), c]
        mask_flatten = torch.cat(mask_flatten, 1) # [bs, sum(h*w across levels)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # [bs, sum(h*w across levels), c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) # [num_levels, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # 得到每张图在token序列上的起始索引
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # [bs, num_levels, 2]

        # 编码器
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten) # 编码器的输出

        # 准备 decoder 的输入
        bs, _, c = memory.shape # bs, token数, 维度
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed 用于 decoder 中，初始 query_embed 维度为 512
            query_embed, tgt = torch.split(query_embed, c, dim=1) # [300, 512] -> [300, 256],[300, 256], 将 query_embed 切分为 query 和 tgt
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1) # [bs, 300, 256], 扩充到对应 batch_size
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1) # [bs, 300, 256]
            reference_points = self.reference_points(query_embed).sigmoid() # [bs, 300, 2], 初始化参考点
            init_reference_out = reference_points
        
        # 解码器
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        '''获取参考点坐标，在多尺度特征图中就是每层的像素的归一化坐标
        :param spatial_shapes: 特征图的尺寸
        :param valid_ratios: 特征图的空洞率
        :param device: 设备
        '''
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 逐层生成参考点坐标
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)) # 生成网格坐标
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 归一化到 [0, 1], 实际最大值会超过1一点,因为不是所有有效区域都是1.0
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1) # 转为 [x, y] 的坐标
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # [batch_size, sum(H_*W_ across levels), 2]
        # 复制4份，每个特征点都有4个归一化参考点, 4个是因为特征点要对应4个采样点，所以就是将每个token对应的参考点复制4份
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # 乘以有效率,将不同层级的参考点统一映射到相同的归一化空间, [bs, h*w across levels, 2] -> [bs, h*w across levels, 4, 2]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        - src: [batch_size, sum(H_*W_ across levels), c], 输入的token序列
        - spatial_shapes: [num_levels, 2], 特征图的尺寸
        - level_start_index: [num_levels], 每张图在token序列上的起始索引
        - valid_ratios: [batch_size, num_levels, 2], 特征图的有效率，用于让query和key尽可能在目标区域
        - pos: [batch_size, sum(H_*W_ across levels), c], 位置编码
        - padding_mask: [batch_size, sum(H_*W_ across levels)], 输入序列的padding掩码
        '''
        output = src
        # 生成参考点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers): # 每一次循环就是一个encoder层，一般 6 个
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # 迭代边界框细化和两阶段可变形DETR的实现
        self.bbox_embed = None
        self.class_embed = None
    
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        '''
        - tgt: [batch_size, 300, 256], 解码器的输入, 预设的输入
        - reference_points: [batch_size, 300, 4, 2], decoder 的参考点坐标
        - src: [batch_size, token_num, c], 编码器的输出
        - src_spatial_shapes: [num_levels, 2], 编码器输出的特征图尺寸
        - src_level_start_index: [num_levels], 编码器输出的每张图在token序列上的起始索引
        - src_valid_ratios: [batch_size, num_levels, 2], 编码器输出的特征图的有效率
        - query_pos: [batch_size, 300, 256], 初始 query 位置
        - src_padding_mask: [batch_size, token_num], 输入序列的padding掩码
        '''
        output = tgt

        intermediate = [] # 中间各层+首尾两层=6层输出的解码结果
        intermediate_reference_points = [] # 中间各层+首尾两层=6层输出的参考点坐标
        for lid, layer in enumerate(self.layers):
            # two stage 下
            if reference_points.shape[-1] == 4:
                pass
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None] # 乘以有效率,将不同层级的参考点统一映射到相同的归一化空间, [bs, 300, 4, 2]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # 迭代边界框细化和两阶段可变形DETR的实现
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # 自注意力
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        '''
        src: [batch_size, sum(H_*W_ across levels), c], 输入的token序列
        pos: [batch_size, sum(H_*W_ across levels), c], 带层位置的位置编码
        reference_points: [batch_size, sum(H_*W_ across levels), 4, 2], 参考点坐标
        spatial_shapes: [num_levels, 2], 特征图的尺寸
        level_start_index: [num_levels], 每张图在token序列上的起始索引
        padding_mask: [batch_size, sum(H_*W_ across levels)], 输入序列的padding掩码
        '''
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2) # 残差连接，将原始输入特征和自注意力输出相加
        src = self.norm1(src) # 层归一化
        src = self.forward_ffn(src) # 前馈网络
        return src


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # 使用多尺度可变形注意力实现交叉注意力
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        '''
        tgt: [batch_size, 300, 256], 解码器的输入
        query_pos: [batch_size, 300, 256], 位置编码
        reference_points: [batch_size, 300, 4, 2], 参考点坐标
        src: [batch_size, token_num, c], 编码器的输出
        src_spatial_shapes: [num_levels, 2], 编码器输出的特征图尺寸
        level_start_index: [num_levels], 编码器输出的每张图在token序列上的起始索引
        src_padding_mask: [batch_size, token_num], 输入序列的padding掩码
        '''
        q = k = self.with_pos_embed(tgt, query_pos) # [bs, 300, 256]
        # 需要转置的原因是 nn.MultiheadAttention 中的参数有 batch_first 这个参数，默认是 False, 所以需要转置
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) # [bs, 300, 256]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 交叉注意力
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask) # [bs, 300, 256]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 前馈网络
        tgt = self.forward_ffn(tgt) # [bs, 300, 256]
        return tgt # [bs, 300, 256]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
        

def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
    )
