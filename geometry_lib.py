import numpy as np
import math
import warnings
import torch
import torch.nn.functional as F
import einops
from einops import rearrange
import torch.nn as nn
from copy import deepcopy
import pytorch3d.ops as p3dops
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid, global_mean_pool
from torch_scatter import segment_csr


############################################
# Utility
############################################

def to_dense_batch(x, batch, return_length=False, input_offset=False):
    input_1d = False
    if len(x.shape) == 1:
        input_1d = True
        x = x[:, None]

    if len(x) == 0:
        a, b = x.reshape(0, *x.shape), torch.zeros([0, 0], dtype=torch.bool, device=x.device)
        if return_length:
            return a, b, torch.zeros([0,], dtype=torch.long, device=x.device)
        else:
            return a, b
    
    if input_offset:
        assert batch[-1].item() == len(x)
        offset = batch
        batch = offset2batch(offset)
    else:
        offset = batch2offset(batch)
    length = offset2length(offset)
    max_n = length.max()
    mask = torch.arange(max_n, device=x.device).reshape(1, -1).repeat(len(offset), 1) < length.view(-1, 1)
    dense_x = torch.zeros((len(offset), max_n, x.shape[-1]), dtype=x.dtype, device=x.device)
    dense_x.view(-1, dense_x.shape[-1])[mask.flatten(), :] = x
    if input_1d:
        dense_x = dense_x[:, :, 0]

    if return_length:
        length = mask.long().sum(1)
        return dense_x, mask, length
    else:
        return dense_x, mask

def to_flat_batch(dense_x, mask):
    if len(dense_x.shape) ==  3:
        x = dense_x.reshape(-1, dense_x.shape[-1])[mask.flatten()]
    else:
        x = dense_x.flatten()[mask.flatten()]
    return x, mask2offset(mask)
    

def batch2mask(batch):
    return to_dense_batch(torch.zeros([len(batch)], device=batch.device), batch)[1]

def offset2mask(offset):
    return batch2mask(offset2batch(offset))

def mask2offset(mask):
    length = mask.sum(dim=1).flatten()
    return length2offset(length)

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else torch.tensor([i] * o) for i, o in enumerate(offset)], dim=0).long().to(offset.device)

def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

def offset2length(offset):
    length = offset.clone()
    length[1:] = offset[1:] - offset[:-1]
    return length

def length2offset(length):
    return torch.cumsum(length, dim=0).long()

def padoffset(offset, L):
    if L > len(offset):
        return torch.cat([offset, torch.full([L - len(offset)], fill_value=offset[-1].item(), device=offset.device)])
    else:
        return offset

o2b = offset2batch
b2o = batch2offset


def split_list_into_groups(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def fallback(*args):
    for a in args:
        if a is not None:
            return a

############################################
# Numpy Utilities
############################################

def order_preserved_unique_np(array, return_inverse=False):
    u, ind, inverse = np.unique(array, return_index=True, return_inverse=True)
    ind = np.argsort(ind)
    u = u[ind]
    if not return_inverse:
        return u
    else:
        for index, value in enumerate(u):
            inverse[array == value] = index
        assert np.all(u[inverse] == array)
        return u, inverse
    

def truncate_top_k(x, k, inplace=False):
    m, n = x.shape
    # get (unsorted) indices of top-k values
    topk_indices = numpy.argpartition(x, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = numpy.indices((m, k))
    kth_vals = x[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = x < kth_vals[:, None]
    # replace mask by 0
    if not inplace:
        return numpy.where(is_smaller_than_kth, 0, x)
    x[is_smaller_than_kth] = 0
    return x    
    
############################################
# Sampling
############################################

def fps_by_sizes(x, offset, sampled_offset, return_points=False, force_cpu=True):
    """more flexible approach
    example: x = (10, 3) tensor, offset=(5,5), sampled_offset=(3,2)
            result = (5, 3) tensor 
    """
    device = x.device
    dense_x, _, length = to_dense_batch(x, offset, return_length=True, input_offset=True)
    sampled_length = offset2length(sampled_offset)
    if force_cpu:
        dense_x, length, sampled_length = dense_x.cpu(), length.cpu(), sampled_length.cpu()
    dense_sample_x, dense_sample_indices = p3dops.sample_farthest_points(dense_x, length, sampled_length)
    if force_cpu:
        dense_sample_x, dense_sample_indices = dense_sample_x.to(device), dense_sample_indices.to(device)
    dense_sample_indices += (torch.cat([torch.zeros([1], device=x.device, dtype=offset.dtype), 
                                        offset[:-1]])).view(-1, 1) 

    dense_sample_x = dense_sample_x.flatten(0, 1)
    dense_sample_indices = dense_sample_indices.flatten()

    sampled_mask = batch2mask(offset2batch(sampled_offset)).flatten()
    dense_sample_x = dense_sample_x[sampled_mask]
    dense_sample_indices = dense_sample_indices[sampled_mask]

    if return_points:
        return dense_sample_x, dense_sample_indices
    else:
        return dense_sample_indices

###########################################
# Reduce
###########################################

def expand(x, index, dim=0):
    """
    can be used as a reverse operation for scatter

    index: 1d index tensor
    Example: 
        x=[[1,2],[3,4]]
        index=[0,0,0,1,1]
        output -> [[1,2],[1,2],[1,2],
                    [3,4],[3,4]]
    """
    expanded_x = torch.index_select(x, dim, index)
    return expanded_x

###########################################
# Neighbor Search
###########################################

def knn(query, base, k, query_offset=None, base_offset=None, pad_offset=False):
    """query: (m, 3), base: (n, 3) -> (m, k) indices, (m, k) dists
    -1 will be returned if the corresponding base has fewer than k points 
    """
    if len(query) == 0:
        return torch.zeros([0, k], device=query.device, dtype=torch.long), torch.zeros([0, k], device=query.device, dtype=torch.float32)
        
    if query_offset is None:
        query_offset = torch.as_tensor([len(base)]).to(base.device)
    if base_offset is None:
        base_offset = query_offset
    if pad_offset and len(base_offset) != len(query_offset):
        if len(base_offset) > len(query_offset):
            query_offset = padoffset(query_offset, len(base_offset))
        else:
            base_offset = padoffset(base_offset, len(query_offset))
    assert len(base_offset) == len(query_offset)
    bs = len(base_offset)

    dense_query, dense_query_mask, query_length = to_dense_batch(query, query_offset, return_length=True, input_offset=True)
    dense_base, _, base_length = to_dense_batch(base, base_offset, return_length=True, input_offset=True)
    dists, idx, _ = p3dops.knn_points(dense_query, dense_base, query_length, base_length, K=k)
    idx += (torch.cat([torch.zeros([1], device=base.device, dtype=base_offset.dtype), 
                        base_offset[:-1]])).view(-1, 1, 1) 
    # first mark -1 and 1e10 for lacking points
    out_of_reach = torch.zeros((bs, k)).bool()
    for i in (base_length < k).nonzero().flatten():
        out_of_reach[i, base_length[i]:] = True

    out_of_reach = out_of_reach.view(bs, 1, k).repeat(1, dense_query.shape[1], 1).flatten()
    idx.view(-1)[out_of_reach] = -1
    dists.view(-1)[out_of_reach] = 1e10

    # flatten
    mask = batch2mask(offset2batch(query_offset)).flatten()
    dists = dists.view(-1, dists.shape[-1])[mask.flatten(), :]
    idx = idx.view(-1, idx.shape[-1])[mask.flatten(), :]
    return idx, dists


def knn_gather(idx, feat, coord=None, with_coord=False, gather_coord=None):
    """
    indexes: (n, k)
    feat: (m, c)
    coord: (m, 3)
    return: (n, k, c), (n, k, 3)
    """
    assert feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c) 
    if with_coord:
        coord = coord.contiguous()            
        # assert coord.is_contiguous()
        coord = torch.cat([coord, torch.zeros([1, 3]).to(coord.device)], dim=0) # for handling -1 indexes
        mask = torch.sign(idx + 1)
        if gather_coord is None:
            gather_coord = coord[:-1]
        grouped_coord = coord[idx.view(-1).long(), :].view(m, nsample, 3) - gather_coord.unsqueeze(1)  # (m, num_sample, 3) normalization
        grouped_coord = torch.einsum(
            "n s c, n s -> n s c", grouped_coord, mask
        )  # (m, num_sample, 3)
        return grouped_feat, grouped_coord
    else:
        return grouped_feat

def resample(query_coord, base_coord, base_feats, k=5, query_offset=None, base_offset=None, reduction='weight_mean', eps=1e-6, return_knn_indexes=True):
    knn_indexes, knn_dists = knn(query_coord, base_coord, k, query_offset=query_offset, base_offset=base_offset)
    knn_base_feats = knn_gather(knn_indexes, base_feats) # (m, k, c)
    if reduction == 'mean':
        v = knn_base_feats.sum(1) / (knn_indexes >= 0).sum(dim=1, keepdim=True)
    elif reduction == 'weight_mean':
        knn_sims = 1 / (knn_dists + eps)
        knn_sims /= knn_sims.sum(dim=1, keepdim=True)
        v = (knn_base_feats * knn_sims[..., None]).sum(1)
    elif reduction == 'max':
        v = knn_base_feats.max(1).values
    elif reduction is None:
        v = knn_base_feats
    if return_knn_indexes:
        return v, knn_indexes
    else:
        return v

###########################################
# Layers 
###########################################
from timm.models.layers import DropPath


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        groups,
        attn_drop_rate=0.0,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
    ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat=None, coord=None, knn_indexes=None, query_feat=None, context_feat=None, context_coord=None): # [156806, 48]
        is_cross = context_feat is not None
        query_feat = fallback(query_feat, feat)
        context_feat = fallback(context_feat, feat)
        context_coord = fallback(context_coord, coord)
        query, key, value = (
            self.linear_q(query_feat),
            self.linear_k(context_feat),
            self.linear_v(context_feat),
        )
        key, pos = knn_gather(knn_indexes, key, context_coord, with_coord=True, gather_coord=coord if is_cross else None) # (torch.Size([156806, 16, 48])
        value = knn_gather(knn_indexes, value, context_coord, with_coord=False) # torch.Size([156806, 16, 48]))
        relation_qk = key - query.unsqueeze(1) # [156806, 16, 48] - [156806, 1, 48] =  [156806, 16, 48]
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias: # position embedding
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk) # [156806, 16, 6])
        weight = self.attn_drop(self.softmax(weight)) # softmax at dim1 [156806, 16, 6])

        mask = torch.sign(knn_indexes + 1)  # all one
        weight = torch.einsum("n s g, n s -> n s g", weight, mask) # [156806, 16, 6], 6 is head
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups) # [156806, 16, 6, 8]
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight) # [156806, 6, 8]
        feat = einops.rearrange(feat, "n g i -> n (g i)") # [156806, 48]
        return feat


class KnnTransformer(nn.Module):
    def __init__(
        self,
        embed_channels,
        n_heads,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(KnnTransformer, self).__init__()
        self.attn = GroupedVectorAttention( 
            embed_channels=embed_channels,
            groups=n_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, feat=None, coord=None, knn_indexes=None, query_feat=None, context_feat=None, context_coord=None):
        assert knn_indexes is not None
        if feat is not None:
            identity = feat
            feat = self.act(self.norm1(self.fc1(feat)))
            feat = (
                self.attn(feat, coord, knn_indexes)
                if not self.enable_checkpoint
                else checkpoint(self.attn, feat, coord, knn_indexes, use_reentrant=True)
            )
        else:
            identity = query_feat
            feat = self.act(self.norm1(self.fc1(query_feat)))
            feat = (
                self.attn(query_feat=feat, coord=coord, context_coord=context_coord, context_feat=context_feat, knn_indexes=knn_indexes)
                if not self.enable_checkpoint
                else checkpoint(self.attn, coord, None, knn_indexes, feat, context_feat, context_coord, use_reentrant=True)
            )
            
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat) # interesting
        feat = self.act(feat)
        return feat 


PointTransformer = KnnTransformer




class DualSoftmaxReposition(nn.Module):
    def __init__(self, hidden_dim, temperature, max_condition_num=-1,
                focal_gamma=2.0, use_projection=True, detach=True, one_way=False):
        super().__init__()
        if use_projection:
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.use_projection = use_projection
        self.temperature = temperature
        self.max_condition_num = max_condition_num
        self.focal_gamma = focal_gamma 
        self.detach = detach
        self.one_way = one_way

    def _detach(self, t):
        if self.detach: return t.detach()
        else: return t
    
        
    def compute_matching_loss(self, conf_matrix, gt_correspondence=None, gt_matrix=None):
        """ conf_matrix: (bsize, N, M)   gt_correspondence: list of (L, 2) """
        if gt_matrix is None:
            assert gt_correspondence is not None
            gt_matrix = DualSoftmaxReposition.to_gt_correspondence_matrix(conf_matrix, gt_correspondence)

        pos_mask = gt_matrix == 1

        if not pos_mask.any():
            warnings.warn("No positive label found in given gt correspondence matrix") 
            return 0.

        conf_matrix = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)
        pos_conf = conf_matrix[pos_mask]
        loss_pos = - torch.pow(1 - pos_conf, self.focal_gamma) * pos_conf.log()
        return loss_pos.mean()
    

    @staticmethod
    def to_gt_correspondence_matrix(conf_matrix, gt_correspondence):
        matrix_gt = torch.zeros_like(conf_matrix)
        for b, match in enumerate (gt_correspondence) :
            matrix_gt[b][match[:, 0], match[:, 1]] = 1
        return matrix_gt


    def match(self, feat_a, coord_a, batch_index_a, feat_b, coord_b, batch_index_b, input_offset=False):
        dev = feat_a.device
        if self.use_projection:
            feat_a, feat_b = self.proj(feat_a), self.proj(feat_b)

        feat_a, mask_a = to_dense_batch(feat_a, batch_index_a, input_offset=input_offset)
        feat_b, mask_b = to_dense_batch(feat_b, batch_index_b, input_offset=input_offset)

        sim_matrix_a2b = torch.einsum("bsc,btc->bst", feat_a, feat_b) / self.temperature

        if self.one_way:
            sim_matrix_a2b.masked_fill_(~mask_a[:, :, None], -100000000)
            sim_matrix_a2b.masked_fill_(~mask_b[:, None, :], float('-inf'))

            softmax_a2b = F.softmax(sim_matrix_a2b, dim=-1)
            return softmax_a2b
        else:
            sim_matrix_b2a = sim_matrix_a2b.clone().transpose(1, 2)

            sim_matrix_a2b.masked_fill_(~mask_a[:, :, None], -100000000)
            sim_matrix_a2b.masked_fill_(~mask_b[:, None, :], float('-inf'))
            sim_matrix_b2a.masked_fill_(~mask_b[:, :, None], -100000000)
            sim_matrix_b2a.masked_fill_(~mask_a[:, None, :], float('-inf'))

            softmax_a2b = F.softmax(sim_matrix_a2b, dim=-1)
            softmax_b2a = F.softmax(sim_matrix_b2a, dim=-1)
            # softmax_a2b = torch.nan_to_num(softmax_a2b, nan=0)
            # softmax_b2a = torch.nan_to_num(softmax_b2a, nan=0)
            conf_matrix = softmax_a2b * softmax_b2a.transpose(1, 2)

            return conf_matrix
        
    def forward(self, feat_a, coord_a, batch_index_a, feat_b, coord_b, batch_index_b, input_offset=False):
        conf_matrix = self.match(feat_a, coord_a, batch_index_a, feat_b, coord_b, batch_index_b, input_offset=input_offset)
        R, t, condition = self.arun(conf_matrix, coord_a, batch_index_a, coord_b, batch_index_b, input_offset=input_offset)
        result = {
            'conf_matrix': conf_matrix,
            'transformation': {
                'condition': condition,
                'R': R, 't': t
            }
        }
        return R, t, result
    
    def arun(self, conf_matrix, coord_a, batch_index_a, coord_b, batch_index_b, input_offset=False):
        bsize, N, M = conf_matrix.shape
        dev = conf_matrix.device
        coord_a, mask_a = to_dense_batch(coord_a, batch_index_a, input_offset=input_offset)
        coord_b, mask_b = to_dense_batch(coord_b, batch_index_b, input_offset=input_offset)

        len_a, len_b = mask_a.sum(dim=1), mask_b.sum(dim=1)
        max_n_entries = torch.cat([len_a[:, None], len_b[:, None]], dim=1).max(dim=1).values.long()
        max_N = max_n_entries.max()
        conf, idx = conf_matrix.view(bsize, -1).sort(descending=True,dim=1)
        sample_mask = offset2mask(length2offset(max_n_entries))
        w = conf[:, :max_N] * sample_mask
        idx = idx[:, :max_N]
        idx_a = idx // M 
        idx_b = idx % M
        b_index = torch.arange(bsize).view(-1, 1).repeat((1, max_N)).flatten().to(dev)
        coord_a_sampled = coord_a[b_index, idx_a.view(-1)].view(bsize, max_N, -1)
        coord_b_sampled = coord_b[b_index, idx_b.view(-1)].view(bsize, max_N, -1)

        try:
            R, t, condition = batch_arun(self._detach(coord_a_sampled), 
                                        self._detach(coord_b_sampled), 
                                        self._detach(w[..., None]))
        except: # fail to get valid solution, this usually happens at the early stage of training
            R = torch.eye(3)[None].repeat(bsize,1,1).type_as(conf_matrix).to(device)
            t = torch.zeros(3, 1)[None].repeat(bsize,1,1).type_as(conf_matrix).to(device)
            condition = torch.zeros(bsize).type_as(conf_matrix)
        
        # filter unreliable solution with condition nnumber
        if self.max_condition_num > 0:
            solution_mask = condition < self.max_condition_num
            R_forwd = R.clone()
            t_forwd = t.clone()
            R_forwd[~solution_mask] = torch.eye(3).type_as(R)
            t_forwd[~solution_mask] = torch.zeros(3, 1).type_as(R)
            R, t = R_forwd, t_forwd
        
        return R, t, condition


def distance_embed(x, temperature = 10000, num_pos_feats=64, scale=10.0):
    x = x[..., None]  # x: [bs, n_dist]
    scale = 2 * math.pi * scale
    dim_t = torch.arange(num_pos_feats)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    sin_x = x * scale / dim_t.to(x.device)
    emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
    return emb


def batch_arun(X, Y, w=None, eps=0.0001):
    '''
    @param X: source frame [B, N, 3]
    @param Y: target frame [B, N, 3]
    @param w: weights [B, N, 1]
    @param eps:
    '''

    bsize = X.shape[0]
    device = X.device
    if w is None:
        w = torch.ones((X.shape[0], X.shape[1], 1), dtype=torch.float32, device=X.device)
    W1 = torch.abs(w).sum(dim=1, keepdim=True)
    w_norm = w / (W1 + eps)
    mean_X = (w_norm * X).sum(dim=1, keepdim=True)
    mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
    Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) )
    Sxy = Sxy.cpu().double()
    U, D, V = Sxy.svd() # small SVD runs faster on cpu
    condition = D.max(dim=1)[0] / D.min(dim=1)[0]
    S = torch.eye(3)[None].repeat(bsize,1,1).double()
    UV_det = U.det() * V.det()
    S[:, 2:3, 2:3] = UV_det.view(-1, 1,1)
    svT = torch.matmul( S, V.transpose(1,2) )
    R = torch.matmul( U, svT).float().to(device)
    t = mean_Y.transpose(1,2) - torch.matmul( R, mean_X.transpose(1,2) )
    return R, t.reshape(-1, 3), condition




class KnnAttentionPool(nn.Module):
    def __init__(self, pool_size, 
                in_channels, out_channels,
                n_heads,
                k,
                qkv_bias=True,
                pe_multiplier=False,
                pe_bias=True,
                reduction='max',
                attn_drop_rate=0.0, bias=False):
        super().__init__()
        self.attn = GroupedVectorAttention( 
            embed_channels=out_channels,
            groups=n_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
        )
        self.pool_size = pool_size
        self.reduction = reduction
        self.k = k
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)        


    def forward(self, coord, feat, offset):
        length = offset2length(offset)
        length[length >= self.pool_size] = self.pool_size
        pooled_offset = length2offset(length)
        feat = self.act(self.norm(self.fc(feat)))
        bs, c = len(offset), feat.shape[-1]
        pooled_indices = fps_by_sizes(coord, offset, pooled_offset)
        pooled_coord = coord[pooled_indices]
        pooled_feat, knn_indexes = resample(pooled_coord, coord, feat, k=self.k, query_offset=pooled_offset, base_offset=offset, return_knn_indexes=True, reduction=self.reduction)
        pooled_feat_attn = self.attn(query_feat=pooled_feat, coord=pooled_coord, context_coord=coord, context_feat=feat, knn_indexes=knn_indexes)
        return pooled_coord, pooled_feat_attn, pooled_offset, pooled_indices



class AttentionPool(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, mask=None): # mask: B, L
        x = x.permute(1, 0, 2)  # B,L,C -> L,B,C
        if mask is None:
            mean_x = x.mean(dim=0, keepdim=True)
        else:
            mean_x = x.sum(dim=0, keepdim=True) / mask.long().sum(dim=1).reshape(1, -1, 1)
            mask = ~mask
            mask = torch.cat([torch.zeros([len(mask), 1], dtype=bool, device=x.device), mask], dim=1)
        x = torch.cat([mean_x, x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            key_padding_mask=mask
        )
        return x.squeeze(0)



class Lambda(nn.Module):
    def __init__(self, lambd, modules={}):
        super().__init__()
        self.lambd = lambd
        if len(modules) > 0:
            self.blocks = nn.ModuleDict(modules)
        else:
            self.blocks = None
    
    def __repr__(self):
        return f'Lambda({self.lambd.__name__})'

    def forward(self, x):
        if self.blocks is not None:
            return self.lambd(x, **self.blocks)
        else:
            return self.lambd(x)
        
        

class PointsEmbedding(nn.Module):
    def __init__(self, output_dim, point_embed_dim=128,
                scale=100., temperature=10000):
        super().__init__()
        self.scale = scale
        self.temperature = temperature
        self.embed_dim = point_embed_dim
        self.linear = nn.Linear(point_embed_dim * 3, output_dim)
    
    def forward(self, points):
        points = points.reshape(-1, 3)
        embed = distance_embed(points, scale=self.scale, temperature=self.temperature, 
                            num_pos_feats=self.embed_dim)
        embed = embed.reshape(-1, 3 * self.embed_dim)
        return self.linear(embed)
    
    
class FrameEmbedding(nn.Module):
    def __init__(self, output_dim, frame_embed_dim=32, frame_size=4, 
                scale=30., temperature=10000):
        super().__init__()
        self.frame_size = frame_size
        self.scale = scale
        self.temperature = temperature
        self.frame_embed_dim = frame_embed_dim
        self.linear = nn.Linear(frame_embed_dim * 3 * frame_size, output_dim)
    
    def forward(self, frames):
        """ frame: (B, frame_size, 3)
        frame_size = 4 (common) or 1 (just t)
        """
        frames = frames.reshape(-1, 3)
        embed = distance_embed(frames, scale=self.scale, temperature=self.temperature, 
                            num_pos_feats=self.frame_embed_dim)
        embed = embed.reshape(-1, self.frame_size * 3 * self.frame_embed_dim)
        return self.linear(embed)

    def embed_points(self, coord, batch_index, frame):
        """coord: (N, 3), batch_index: (N), frame: (B, 4, 3)"""
        dense_coord, mask = to_dense_batch(coord, batch_index) # (B, L, 3)
        B, L = mask.shape
        dense_coord = dense_coord[:, :, None, :] # (B, L, 1, 3)
        frame = frame[:, None, :, :] # (B, 1, 4, 3)
        relative_coord = dense_coord - frame # (B, L, 4, 3)
        relative_emb = self(relative_coord).view(B, L, -1) # (B*L, emb)
        relative_emb, _ = to_flat_batch(relative_emb, mask)
        return relative_emb


##############################################


def batch_Rt_transform(pts, R, t):
    """ pts: (B, N, 3), R: (B, 3, 3), t: (B, 3) """
    return torch.bmm(R, pts.transpose(1, 2)).transpose(1,2) + t.view(-1, 1, 3)


def batch_X_transform_flat(flat_pts, batch_index, X, input_offset=False):
    return batch_Rt_transform_flat(flat_pts, batch_index, *batch_X_to_Rt(X), input_offset=input_offset)

def batch_Rt_transform_flat(flat_pts, batch_index, R, t, input_offset=False):
    dense_pts, mask = to_dense_batch(flat_pts, batch_index, input_offset=input_offset)
    assert len(dense_pts) == len(R)
    dense_pts = batch_Rt_transform(dense_pts, R, t)
    flat_pts, _ = to_flat_batch(dense_pts, mask)
    return flat_pts

def batch_X_to_Rt(X):
    return X[:, :3, :3], X[:, :3, -1]






class BaseTransformerNetwork(nn.Module):
    NON_MODULE_LAYERS = ['add_embedding']

    def __init__(self, layer_types, blocks, skip_repo=False):
        super().__init__()
        self.skip_repo = skip_repo
        self.layer_types = layer_types
        assert len([l for l in layer_types if not any([l.startswith(el) for el in self.NON_MODULE_LAYERS])]) == len(blocks)
        blocks = [nn.ModuleList(blk) if isinstance(blk, (list, tuple)) else blk
                    for blk in blocks]
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def parse_layer_name(layer):
        return layer.split('(')[0]

    @staticmethod
    def parse_interaction_roles(layer):
        return [a.strip() for a in layer.split('(')[1].replace(')', '').split(',')]

    @staticmethod
    def as_mod_list(mod):
        if isinstance(mod, nn.ModuleList): return mod
        else: return [mod]
    
    @staticmethod
    def parse_feat_name(name):
        if '[' in name:
            return name, name.split('[')[0]
        else:
            return name, name


class KnnTransformerNetwork(BaseTransformerNetwork):
    def forward(self, feat={}, coord={}, batch_index={}, knn_indexes={}, embedding={}, collect_cross=False):
        layer_id = 0 
        position_layer_outputs = []
        cross_result = []
        for layer in self.layer_types:
            layer_name = self.parse_layer_name(layer)
            if layer_name == 'self':
                knn_t_block = self.as_mod_list(self.blocks[layer_id])
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    kn, n = self.parse_feat_name(n)
                    feat[n] = knn_t_block[i](feat[n], coord[n], knn_indexes[f'{kn}2{kn}'])
            elif layer_name == 'cross':
                knn_t_block = self.as_mod_list(self.blocks[layer_id])
                tmp_result = {}
                for block, (a, b) in zip(knn_t_block, split_list_into_groups(self.parse_interaction_roles(layer), 2)):
                    ka, a = self.parse_feat_name(a)
                    kb, b = self.parse_feat_name(b)
                    tmp_result[a] = block(query_feat=feat[a], coord=coord[a], context_feat=feat[b], context_coord=coord[b], knn_indexes=knn_indexes[f'{ka}2{kb}'])
                feat.update(tmp_result)
                if collect_cross:
                    cross_result.append(tmp_result)
            elif layer_name == 'positioning':
                a, b = self.parse_interaction_roles(layer)
                reposition_block = self.blocks[layer_id]
                embedding_block = None
                if isinstance(reposition_block, nn.ModuleList):
                    reposition_block, embedding_block = reposition_block[0], reposition_block[1]
                R, t, others = reposition_block(feat[a], coord[a], batch_index[a], feat[b], coord[b], batch_index[b])
                position_layer_outputs.append({
                    'R': R,
                    't': t,
                    **others
                })
                coord_a, coord_b, feat_a, feat_b = coord[a], coord[b], feat[a], feat[b]
                if not self.skip_repo:
                    coord_a = batch_Rt_transform_flat(coord_a, batch_index[a], R, t)
                    k_a2b, k_b2a = knn_indexes[f'{a}2{b}'].shape[-1], knn_indexes[f'{b}2{a}'].shape[-1]
                    knn_indexes[f'{a}2{b}'], _ = knn(coord_a, coord_b, k_a2b, query_offset=batch2offset(batch_index[a]), base_offset=batch2offset(batch_index[b]))
                    knn_indexes[f'{b}2{a}'], _ = knn(coord_b, coord_a, k_b2a, query_offset=batch2offset(batch_index[b]), base_offset=batch2offset(batch_index[a]))
                if embedding_block is not None:
                    feat_a = feat_a + embedding_block(coord_a)
                    feat_b = feat_b + embedding_block(coord_b)
                coord[a], feat[a], feat[b] = coord_a, feat_a, feat_b
            elif layer_name == 'add_embedding':
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    if isinstance(embedding[n], list):
                        feat[n] = feat[n] + embedding[n].pop(0)
                    else:
                        feat[n] = feat[n] + embedding[n]
            elif "lambda" in layer_name:
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    kn, n = self.parse_feat_name(n)
                    feat[n] = self.blocks[layer_id](feat[n])
            else:
                raise KeyError(layer)
            if layer_name not in self.NON_MODULE_LAYERS: layer_id += 1
        
        if len(position_layer_outputs) > 0:
            return coord, feat, knn_indexes, position_layer_outputs
        else:
            if collect_cross:
                return feat, cross_result
            else:
                return feat
        
        
class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        hidden_dim=512,
        dropout=0.0,
        cross=False
    ):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.pre_norm_ff = nn.LayerNorm(d_model)
        self.is_cross = cross
        if cross:
            self.pre_norm_attn_q = nn.LayerNorm(d_model)
            self.pre_norm_attn_c = nn.LayerNorm(d_model)
        else:
            self.pre_norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query,
        context=None,
        query_mask=None,
        context_mask=None
    ):
        if self.is_cross:
            query, context = self.pre_norm_attn_q(query), self.pre_norm_attn_c(context)
        else:
            query = self.pre_norm_attn(query)

        if context is None:
            context = query
            context_mask = query_mask        
        
        kp_mask = None 
        if context_mask is not None: kp_mask = ~context_mask
        
        attn_query = self.self_attn(query.transpose(0, 1), context.transpose(0, 1), value=context.transpose(0, 1), 
                                    key_padding_mask=kp_mask)[0].transpose(0, 1)
        attn_query = torch.nan_to_num(attn_query)
        query = attn_query + query
        query = self.pre_norm_ff(query)
        query = self.ff(query) + query
        if query_mask is not None:  query = query * query_mask[:, :, None]
        return query

        
        
class TransformerNetwork(BaseTransformerNetwork):
    def forward(self, feat={}, mask={}, embedding={}, collect={}):
        layer_id = 0 
        collect_result = {k: [] for k in collect}
        collect_result['layer_names'] = {k: [] for k in collect}
        for layer in self.layer_types:
            layer_name = self.parse_layer_name(layer)
            if layer_name == 'self':
                t_blocks = self.as_mod_list(self.blocks[layer_id])
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    kn, n = self.parse_feat_name(n)
                    feat[n] = t_blocks[i](query=feat[n], query_mask=mask[n])     # TODO
            elif layer_name == 'cross':
                t_blocks = self.as_mod_list(self.blocks[layer_id])
                tmp_result = {}
                for block, (a, b) in zip(t_blocks, split_list_into_groups(self.parse_interaction_roles(layer), 2)):
                    ka, a = self.parse_feat_name(a)
                    kb, b = self.parse_feat_name(b)
                    tmp_result[a] = block(query=feat[a], context=feat[b], query_mask=mask.get(a, None), context_mask=mask.get(b, None))
                feat.update(tmp_result)
            elif layer_name == 'add_embedding':
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    feat[n] = feat[n] + embedding[n].pop(0)
            elif "lambda" in layer_name:
                for i,n in enumerate(self.parse_interaction_roles(layer)):
                    kn, n = self.parse_feat_name(n)
                    feat[n] = self.blocks[layer_id](feat[n])
            for k, v in collect.items():
                if layer_id in v:
                    collect_result[k].append(feat[k])
                    collect_result['layer_names'][k].append(layer)
            if layer_name not in self.NON_MODULE_LAYERS: layer_id += 1
        if len(collect) > 0:
            return feat, collect_result
        else:
            return feat




class VoxelPooling(nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(VoxelPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, skip_fc=False, **point_attributes):
        coord, feat, offset = points
        batch = offset2batch(offset) # [0000...1111]
        if not skip_fc: feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]), # [     0,  76806, 156806]
                reduce="min") # [2, 3]
        cluster = voxel_grid( # torch_geometric
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0 # grid_size = 0.1
        )
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster, stable=True)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)]) # [     0,      3,      9,  ..., 156796, 156800, 156806]
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean") # pooling
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max") # the segment csr and voxel grid is the key operation for pooling
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        # segment_csr(flow[indices], idx_ptr, reduce='mean')
        out_point_attributes = {}
        if len(point_attributes) > 0:
            for k, v in point_attributes.items():
                if k in ['cluster', 'indices', 'idx_ptr']: continue
                reduce = 'max'
                if isinstance(v, (list, tuple)):
                    reduce, v = v
                prev_bool = False
                if v.dtype == torch.bool:
                    prev_bool = True
                    v = v.long()
                elif v.dtype in [torch.float32, torch.float64]:
                    reduce = 'mean'
                out_point_attributes[k] = segment_csr(v[sorted_cluster_indices], idx_ptr, reduce=reduce)
                if prev_bool: v = v.bool()
        
        # to transform coordinate, just `cluster[corr[:, 0]]`
        return [coord, feat, offset], {'cluster': cluster, 'indices': sorted_cluster_indices, 'idx_ptr': idx_ptr, **out_point_attributes}


##########################################################################################################



class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = self.proj(feat)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class PointTransformerSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(PointTransformerSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0.0 for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PointTransformer(
                embed_channels=embed_channels,
                n_heads=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint,
            )
            self.blocks.append(block)

    def forward(self, points, return_knn_indexes=False):
        coord, feat, offset = points
        knn_index, _ = knn(coord, coord, self.neighbours, query_offset=offset)
        for block in self.blocks:
            feat = block(feat, coord, knn_index)
        if return_knn_indexes:
            return [coord, feat, offset], knn_index
        else:
            return [coord, feat, offset]


class PointPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        enable_checkpoint=False,
    ):
        super(PointPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = PointTransformerSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat) # just linear, [156806, 48]
        return self.blocks([coord, feat, offset])


class PointEncoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        groups,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
    ):
        super(PointEncoder, self).__init__()

        self.down = VoxelPooling(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = PointTransformerSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, return_knn_indexes=False, **point_attributes):
        points, cluster = self.down(points, **point_attributes)
        return self.blocks(points, return_knn_indexes=return_knn_indexes), cluster


class PointDecoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        skip_channels,
        embed_channels,
        groups,
        neighbours=16,
        qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=None,
        drop_path_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
    ):
        super(PointDecoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = PointTransformerSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class PointTransformerNetwork(nn.Module):
    def __init__(self, grid_sizes,  # (2)
                depths,  # (3)
                dec_depths, # (2)
                hidden_dims, # (3)
                n_heads, # (3) 
                ks, # (3)
                in_dim=None, skip_dec=False):
        super().__init__()
        self.skip_dec = skip_dec
        self.enc_stages = nn.ModuleList()
        if not skip_dec:
            self.dec_stages = nn.ModuleList()

        self.patch_embed = PointPatchEmbed(
            in_channels=in_dim or hidden_dims[0],
            embed_channels=hidden_dims[0],
            groups=n_heads[0],
            depth=depths[0],
            neighbours=ks[0])

        for i in range(len(depths) - 1):
            self.enc_stages.append(
                PointEncoder(
                    depth=depths[i+1],
                    in_channels=hidden_dims[i],
                    embed_channels=hidden_dims[i + 1],
                    groups=n_heads[i+1],
                    grid_size=grid_sizes[i],
                    neighbours=ks[i+1]))
            
            if not skip_dec:
                self.dec_stages.insert(0, 
                    PointDecoder(
                        depth=dec_depths[i],
                        in_channels=hidden_dims[i+1],
                        skip_channels=hidden_dims[i],
                        embed_channels=hidden_dims[i],
                        groups=n_heads[i],
                        neighbours=ks[i])
                )
            
    def forward(self, points, return_full=False):
        points = self.patch_embed(points)
        cluster_indexes = []
        all_points = [points]
        for i, stage in enumerate(self.enc_stages):
            points, attrs = stage(points)
            cluster_indexes.insert(0, attrs['cluster'])
            all_points.insert(0, points)

        if not self.skip_dec:
            for i, dec_stage in enumerate(self.dec_stages):
                points, skip_points = all_points[i], all_points[i+1]
                cluster = cluster_indexes[i]
                points = dec_stage(points, skip_points, cluster)
                all_points[i+1] = points
        
        if return_full:
            return points, all_points, cluster_indexes
        else:
            return points


##########################################################################################################


def make_knn_transformer_one_layer(l, hidden_dim, n_heads):
    knn_t_block = KnnTransformer(hidden_dim, n_heads)
    if 'self' in l:
        out = [deepcopy(knn_t_block) for _ in range(l.count(',') + 1)]
    elif 'cross' in l:
        if l.count(',') == 1:
            out = deepcopy(knn_t_block)
        else:
            assert l.count(',') == 3
            out = [deepcopy(knn_t_block), deepcopy(knn_t_block)]
    else:
        raise NotImplementedError()
    return out
    
    
def make_knn_transformer_layers(layers, hidden_dim, n_heads):
    out_layers = []
    for l in layers:
        assert l.count('(') == 1
        if 'self' in l or 'cross' in l:
            out_layers.append(make_knn_transformer_one_layer(l, hidden_dim, n_heads))
        elif 'add_embedding' in l:
            continue
        else:
            raise ValueError()
    return out_layers


def make_transformer_layers(layers, input_dim, n_heads, hidden_ff_dim, dropout=0.0):
    out_layers = []
    t_block = TransformerLayer(input_dim, n_heads, hidden_ff_dim, dropout)
    t_block_cross = TransformerLayer(input_dim, n_heads, hidden_ff_dim, dropout, cross=True)
    for l in layers:
        assert l.count('(') == 1
        if 'self' in l:
            out_layers.append([deepcopy(t_block) for _ in range(l.count(',') + 1)])
        elif 'cross' in l:
            if l.count(',') == 1:
                out_layers.append(deepcopy(t_block_cross))
            else:
                num_layers = (l.count(',') + 1) // 2
                out_layers.append([deepcopy(t_block_cross) for _ in range(num_layers)])
        else:
            raise ValueError()
    return out_layers



    