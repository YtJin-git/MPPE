import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
from model.alpha_clip import *

class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, ):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q


class MPPE(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        # self.clip = load_clip(name="E:\ZSLSampleProjects\GAN-CZSL\pretrained_clip\ViT-L-14.pt", context_length=config.context_length)
        self.alpha_clip_model, preprocess = alpha_clip.load(name=config.clip_arch,
                                                       alpha_vision_ckpt_pth=config.alpha_vision_ckpt_pth,
                                                       context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0.1
        self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.alpha_clip_model.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.alpha_clip_model, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.alpha_clip_model.visual.output_dim

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)

        # context prompt attention module
        self.cam = nn.ModuleList([CrossAttentionLayer(output_dim, output_dim // 64, self.cross_attn_dropout) for _ in
                                  range(config.cmt_layers)])
        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.alpha_clip_model.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.alpha_clip_model.visual.transformer.width,
                                        bottleneck=self.config.adapter_dim,
                                        dropout=self.config.adapter_dropout
                                        ) for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor, alpha=None):
        return self.encode_image_with_adapter(x, alpha)

    def encode_image_with_adapter(self, x: torch.Tensor, alpha=None):
        x = self.alpha_clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x + self.alpha_clip_model.visual.conv1_alpha(alpha)    # 加入alpha分支
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.alpha_clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                  dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.alpha_clip_model.visual.positional_embedding.to(x.dtype)
        x = self.alpha_clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2).type(torch.float32)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.alpha_clip_model.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.alpha_clip_model.visual.transformer.resblocks[i_block].attention(
                self.alpha_clip_model.visual.transformer.resblocks[i_block].ln_1(x.type(self.alpha_clip_model.dtype))
            )[0]
            x = x + adapt_x + residual
            # x = x + residual

            # FFN
            i_adapter = i_block + self.alpha_clip_model.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.alpha_clip_model.visual.transformer.resblocks[i_block].mlp(
                self.alpha_clip_model.visual.transformer.resblocks[i_block].ln_2(x.type(self.alpha_clip_model.dtype))
            )
            x = x + adapt_x + residual
            # x = x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.alpha_clip_model.visual.ln_post(img_feature).type(self.alpha_clip_model.dtype)
        if self.alpha_clip_model.visual.proj is not None:
            img_feature = img_feature @ self.alpha_clip_model.visual.proj
        return img_feature[:, 0, :], img_feature

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.alpha_clip_model.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.alpha_clip_model.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.alpha_clip_model.dtype)
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.alpha_clip_model.dtype)
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.alpha_clip_model.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def construct_token_tensors(self, pair_idx, bias=None):
        if bias is None:
            attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
            token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
            for i_element in range(self.token_ids.shape[0]):
                class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
                token_tensor.append(self.alpha_clip_model.token_embedding(
                    class_token_ids.cuda()
                ).type(self.alpha_clip_model.dtype))

            eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
            soft_att_obj = self.attr_dropout(self.soft_att_obj)
            # comp
            token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
                attr_idx
            ].type(self.alpha_clip_model.dtype)
            token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
                obj_idx + self.offset
                ].type(self.alpha_clip_model.dtype)
            token_tensor[0][
                :, 1: len(self.comp_ctx_vectors) + 1, :
            ] = self.comp_ctx_vectors.type(self.alpha_clip_model.dtype)
            # attr
            token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
                                                    :self.offset
                                                    ].type(self.alpha_clip_model.dtype)
            token_tensor[1][
                :, 1: len(self.attr_ctx_vectors) + 1, :
            ] = self.attr_ctx_vectors.type(self.alpha_clip_model.dtype)
            # obj
            token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
                                                    self.offset:
                                                    ].type(self.alpha_clip_model.dtype)
            token_tensor[2][
                :, 1: len(self.obj_ctx_vectors) + 1, :
            ] = self.obj_ctx_vectors.type(self.alpha_clip_model.dtype)

        else:
            attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
            token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
            for i_element in range(self.token_ids.shape[0]):
                class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
                token_tensor.append(self.alpha_clip_model.token_embedding(
                    class_token_ids.cuda()
                ).type(self.alpha_clip_model.dtype))

            eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
            soft_att_obj = self.attr_dropout(self.soft_att_obj)

            # comp
            token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
                attr_idx
            ].type(self.alpha_clip_model.dtype)
            token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
                obj_idx + self.offset
                ].type(self.alpha_clip_model.dtype)

            ctx_c = self.comp_ctx_vectors.type(self.alpha_clip_model.dtype).unsqueeze(0)  # (1, n_ctx, ctx_dim)
            bias_c = bias[0].unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx_shifted_c = ctx_c + bias_c  # (batch, n_ctx, ctx_dim)

            ctx_shifted_c = ctx_shifted_c / ctx_shifted_c.norm(dim=-1, keepdim=True)

            ctx_shifted_c = ctx_shifted_c.unsqueeze(1).expand(-1, token_tensor[0].shape[0], -1, -1)
            token_tensor_batch_c = token_tensor[0].expand(bias_c.shape[0], -1, -1, -1).clone()
            token_tensor_batch_c[:, :, 1: len(self.comp_ctx_vectors) + 1, :] = ctx_shifted_c
            token_tensor[0] = token_tensor_batch_c.type(self.alpha_clip_model.dtype)

            # attr
            token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
                                                    :self.offset
                                                    ].type(self.alpha_clip_model.dtype)
            ctx_a = self.attr_ctx_vectors.type(self.alpha_clip_model.dtype).unsqueeze(0)  # (1, n_ctx, ctx_dim)
            bias_a = bias[1].unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx_shifted_a = ctx_a + bias_a  # (batch, n_ctx, ctx_dim)

            ctx_shifted_a = ctx_shifted_a / ctx_shifted_a.norm(dim=-1, keepdim=True)

            ctx_shifted_a = ctx_shifted_a.unsqueeze(1).expand(-1, token_tensor[1].shape[0], -1, -1)
            token_tensor_batch_a = token_tensor[1].expand(bias_a.shape[0], -1, -1, -1).clone()
            token_tensor_batch_a[:, :, 1: len(self.attr_ctx_vectors) + 1, :] = ctx_shifted_a
            token_tensor[1] = token_tensor_batch_a.type(self.alpha_clip_model.dtype)

            # obj
            token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
                                                    self.offset:
                                                    ].type(self.alpha_clip_model.dtype)
            ctx_o = self.obj_ctx_vectors.type(self.alpha_clip_model.dtype).unsqueeze(0)  # (1, n_ctx, ctx_dim)
            bias_o = bias[2].unsqueeze(1)  # (batch, 1, ctx_dim)
            ctx_shifted_o = ctx_o + bias_o  # (batch, n_ctx, ctx_dim)

            ctx_shifted_o = ctx_shifted_o / ctx_shifted_o.norm(dim=-1, keepdim=True)

            ctx_shifted_o = ctx_shifted_o.unsqueeze(1).expand(-1, token_tensor[2].shape[0], -1, -1)
            token_tensor_batch_o = token_tensor[2].expand(bias_o.shape[0], -1, -1, -1).clone()
            token_tensor_batch_o[:, :, 1: len(self.obj_ctx_vectors) + 1, :] = ctx_shifted_o
            token_tensor[2] = token_tensor_batch_o.type(self.alpha_clip_model.dtype)
        return token_tensor

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, _mask, batch_attr, batch_obj, batch_target = target
        if len(predict) == 5:
            comp_logits, attr_logits, obj_logits, comp_cam_a2o_logits, comp_cam_o2a_logits = predict
        else:
            comp_logits, attr_logits, obj_logits, comp_cam_a2o_logits = predict
            comp_cam_o2a_logits = comp_cam_a2o_logits
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        loss_comp_cam_a2o = loss_fn(comp_cam_a2o_logits, batch_target)
        loss_comp_cam_o2a = loss_fn(comp_cam_o2a_logits, batch_target)
        loss = loss_comp * self.config.pair_loss_weight + \
               loss_attr * self.config.attr_loss_weight + \
               loss_obj * self.config.obj_loss_weight + \
               loss_comp_cam_a2o * self.config.pair_cam_a2o_loss_weight + \
               loss_comp_cam_o2a * self.config.pair_cam_o2a_loss_weight
        return loss

    def logit_infer(self, predict, pairs):
        if len(predict)==5:
            comp_logits, attr_logits, obj_logits, comp_cam_a2o_logits, comp_cam_o2a_logits = predict
            comp_logits = (comp_logits + comp_cam_a2o_logits * self.config.pair_cam_a2o_inference_weight +
                           comp_cam_o2a_logits * self.config.pair_cam_o2a_inference_weight)
        else:
            comp_logits, attr_logits, obj_logits, comp_cam_a2o_logits = predict
            comp_logits = (comp_logits + comp_cam_a2o_logits * self.config.pair_cam_a2o_inference_weight)
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][
                                                                                                   0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][
                                                                                                1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:,
                                     i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        return comp_logits

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        text_features_c_a2o = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            if i_element == 1:
                text_features_a = idx_text_features
            elif i_element == 2:
                text_features_o = idx_text_features
            text_features.append(idx_text_features)
            del idx_text_features

        for i in range(idx.shape[0]):
            text_feature_a = text_features_a[idx[i, 0]]
            text_feature_o = text_features_o[idx[i, 1]]

            text_feature_a = text_feature_a.unsqueeze(0).expand(1, -1, -1).type(torch.float32)
            text_feature_o = text_feature_o.unsqueeze(0).expand(1, -1, -1).type(torch.float32)

            # a2o
            text_feature_cam = text_feature_o
            for layer in self.cam:
                text_feature_cam = layer(text_feature_cam, text_feature_a)

            text_feature_cam = text_feature_o.squeeze() + self.lamda * text_feature_cam.squeeze()
            # text_feature_cam = text_feature_o.squeeze() + text_feature_cam.squeeze()
            text_features_c_a2o.append(text_feature_cam)

        text_features_c_a2o = torch.stack(text_features_c_a2o).type(self.dtype)
        text_features_c_a2o = text_features_c_a2o / text_features_c_a2o.norm(dim=-1, keepdim=True)
        text_features.append(text_features_c_a2o)
        return text_features

    def forward_for_open(self, batch, text_feats):
        batch_img, batch_mask = batch[0].cuda(), batch[1].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.dtype), batch_mask.type(self.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img.type(torch.float32)).type(self.dtype),
                              self.obj_disentangler(batch_img.type(torch.float32)).type(self.dtype)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]

            logits.append(
                torch.einsum(
                    "bd, kd->bk",
                    normalized_img_features[i_element],
                    idx_text_features * self.alpha_clip_model.logit_scale.exp()
                ))
        logits.append(
            torch.einsum(
                "bd, kd->bk",
                normalized_img_features[0],
                text_feats[-1] * self.alpha_clip_model.logit_scale.exp()
            ))
        return logits

    def forward(self, batch, idx):
        batch_img, batch_mask = batch[0].cuda(), batch[1].cuda()
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.dtype), batch_mask.type(self.dtype))
        # ========================此处proj
        batch_img_features = [batch_img, self.attr_disentangler(batch_img.type(torch.float32)).type(self.dtype),
                              self.obj_disentangler(batch_img.type(torch.float32)).type(self.dtype)]

        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]


        # bias = self.meta_net(torch.stack(batch_img_features))   # (3, batch, ctx_dim)

        # batch_img_features_tensor = torch.stack(batch_img_features)
        # bias_c = self.meta_net_c(batch_img_features_tensor[0])
        # bias_a = self.meta_net_a(batch_img_features_tensor[1])
        # bias_o = self.meta_net_o(batch_img_features_tensor[2])
        # bias = [bias_c, bias_a, bias_o]
        bias = None
        token_tensors = self.construct_token_tensors(idx, bias)

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            if bias is None:
                logits.append(
                    torch.einsum(
                        "bd, kd->bk",
                        normalized_img_features[i_element],
                        idx_text_features * self.alpha_clip_model.logit_scale.exp()
                ))
            else:
                logits.append(
                    torch.einsum(
                        "bd, bkd->bk",
                        normalized_img_features[i_element],
                        idx_text_features * self.alpha_clip_model.logit_scale.exp()
                    ))

            if i_element == 1:
                text_features_a = idx_text_features
            elif i_element == 2:
                text_features_o = idx_text_features
            else:
                text_features_c = idx_text_features

        text_features_c_a2o = list()  # q为text_feature_o，kv为text_feature_a
        text_features_c_o2a = list()
        if bias is None:
            for i in range(idx.shape[0]):
                text_feature_a = text_features_a[idx[i, 0]]
                text_feature_o = text_features_o[idx[i, 1]]

                text_feature_a = text_feature_a.unsqueeze(0).expand(1, -1, -1).type(torch.float32)
                text_feature_o = text_feature_o.unsqueeze(0).expand(1, -1, -1).type(torch.float32)

                # a2o
                text_feature_cam = text_feature_o
                for layer in self.cam:
                    text_feature_cam = layer(text_feature_cam, text_feature_a)

                text_feature_cam = text_feature_o.squeeze() + self.lamda * text_feature_cam.squeeze()
                # text_feature_cam = text_feature_o.squeeze() + text_feature_cam.squeeze()
                text_features_c_a2o.append(text_feature_cam)

                # o2a
                # text_feature_cam = text_feature_a
                # for layer in self.cam:
                #     text_feature_cam = layer(text_feature_cam, text_feature_o)
                #
                # text_feature_cam = text_feature_a.squeeze() + self.lamda * text_feature_cam.squeeze()
                # text_features_c_o2a.append(text_feature_cam)


            text_features_c_a2o = torch.stack(text_features_c_a2o).type(self.dtype)
            text_features_c_a2o = text_features_c_a2o / text_features_c_a2o.norm(dim=-1, keepdim=True)
            logits.append(
                torch.einsum(
                    "bd, kd->bk",
                    normalized_img_features[0],
                    text_features_c_a2o * self.alpha_clip_model.logit_scale.exp()
                ))
            logits.append(torch.zeros(b, idx.shape[0]).cuda())

        else:   # bias is not None
            for i in range(idx.shape[0]):
                text_feature_a = text_features_a[:, idx[i, 0], :]
                text_feature_o = text_features_o[:, idx[i, 1], :]

                text_feature_a = text_feature_a.unsqueeze(1).type(torch.float32)
                text_feature_o = text_feature_o.unsqueeze(1).type(torch.float32)

                # a2o
                text_feature_cam = text_feature_o
                for layer in self.cam:
                    text_feature_cam = layer(text_feature_cam, text_feature_a)

                text_feature_cam = text_feature_o.squeeze() + self.lamda * text_feature_cam.squeeze()
                text_features_c_a2o.append(text_feature_cam)


            text_features_c_a2o = torch.stack(text_features_c_a2o).permute(1, 0, 2).type(self.dtype)
            text_features_c_a2o = text_features_c_a2o / text_features_c_a2o.norm(dim=-1, keepdim=True)
            logits.append(
                torch.einsum(
                    "bd, bkd->bk",
                    normalized_img_features[0],
                    text_features_c_a2o * self.alpha_clip_model.logit_scale.exp()
                ))
            logits.append(torch.zeros(b, idx.shape[0]).cuda())

        return logits

