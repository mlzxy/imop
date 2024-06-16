import torch.nn as nn
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy as dc
import geometry_lib as glib
from utils.object import Section
from torch_geometric.nn.pool import global_mean_pool


class InvariantRegionNetwork(nn.Module):
    def __init__(self, reason_depth=4):
        super().__init__()
        self.backbone = glib.PointTransformerNetwork(grid_sizes=(0.015, 0.03), 
                                                depths=(2, 4, 2), 
                                                dec_depths=None, 
                                                hidden_dims=(64, 128, 256), 
                                                n_heads=(4, 8, 8), ks=(16, 24, 32), in_dim=14, skip_dec=True)
        self.reason_stages = nn.ModuleList()
        layers = ['cross(a, b, b, a)', 'self(a, b)', 'cross(a, b, b, a)']
        self.key_linears = nn.ModuleList()
        for i in range(reason_depth):
            _layers = dc(layers)
            if i == reason_depth - 1:
                _layers[-1] = 'cross(a, b)'

            block = glib.KnnTransformerNetwork(_layers, glib.make_knn_transformer_layers(_layers, 256, 8))
            self.reason_stages.append(block)
            self.key_linears.append(nn.Linear(256, 1))
        
        self.temperature = 0.25
        self.k = 16


    def forward(self, batch, debug=False):
        bsize = len(batch['src']['t']['X_to_robot_frame']) 
        dev = batch['src']['t']['pcd'].device
        output, loss_dict, metric_dict = defaultdict(list), {}, {}
        t1, t2 = batch['src']['t'], batch['src']['t+1']

        for item in [t1, t2]:
            item['offset'] = glib.batch2offset(item['batch_index'])
            item['robot_pcd'] = glib.batch_X_transform_flat(item['pcd'], item['batch_index'], item['X_to_robot_frame'])

            item['feat'] = torch.cat([item[k] for k in ['pcd', 'rgb', 'normal', 'robot_pcd']], dim=1)
            item['open'], item['ignore_col'] = item['open'].reshape(-1, 1).float(), item['ignore_col'].reshape(-1, 1).float()
            item['feat'] = torch.cat([item['feat'],
                                        glib.expand(item['open'], item['batch_index']),
                                        glib.expand(item['ignore_col'], item['batch_index'])], dim=1)       

        for item in [t1, t2]:
            item['coarse_pcd'], item['coarse_feat'], item['coarse_offset'] = self.backbone([item['pcd'], item['feat'], item['offset']])

        if self.training or debug:
            t1['coarse_key_mask'] = self.get_coarse_mask(t1['pcd'], t1['offset'], t1['coarse_pcd'], t1['coarse_offset'], 
                                                        key_mask=t1['key_mask'])['key_mask']
        
        knn_indexes = {
            'a2a': glib.knn(t1['coarse_pcd'], t1['coarse_pcd'], self.k, query_offset=t1['coarse_offset'])[0],
            'b2b': glib.knn(t2['coarse_pcd'], t2['coarse_pcd'], self.k, query_offset=t2['coarse_offset'])[0],
            'a2b': glib.knn(t1['coarse_pcd'], t2['coarse_pcd'], self.k, query_offset=t1['coarse_offset'], base_offset=t2['coarse_offset'])[0],
            'b2a': glib.knn(t2['coarse_pcd'], t1['coarse_pcd'], self.k, query_offset=t2['coarse_offset'], base_offset=t1['coarse_offset'])[0]
        }
        
        for i, stage in enumerate(self.reason_stages):
            tmp = stage(feat={'a': t1['coarse_feat'], 'b': t2['coarse_feat']}, 
                        coord={'a': t1['coarse_pcd'], 'b': t2['coarse_pcd']}, 
                        knn_indexes=knn_indexes)
            t1['coarse_feat'], t2['coarse_feat'] = tmp['a'], tmp['b']
            logits = self.key_linears[i](t1['coarse_feat']) / self.temperature
            output['coarse_prob_map'].append(logits.sigmoid())

            if self.training or debug:
                _loss_dict, _metric_dict = self.get_loss(logits.squeeze(-1), t1['coarse_key_mask'], t1['coarse_offset'])
                loss_dict.update({f'{k}{i}': v for k, v in _loss_dict.items()})
                metric_dict.update({f'{k}{i}': v for k, v in _metric_dict.items()})
        
        if not self.training:
            prob_map = self.to_fine_map(t1['pcd'], t1['offset'], t1['coarse_pcd'], t1['coarse_offset'], 
                                                        coarse_prob_map=output['coarse_prob_map'][-1])['coarse_prob_map']
            output['prob_map'] = prob_map
    
        return {'output': dict(output), 
                'loss_dict': loss_dict, 'metric_dict': metric_dict}

    def compute_iou(self, inputs, gt_mask, batch_mask):
        pred_mask = inputs.sigmoid() > 0.5
        pred_mask *= batch_mask
        iou = (2 * (pred_mask * gt_mask).sum(1)) / (pred_mask.sum(1) + gt_mask.sum(1) + 1e-3)
        return iou.mean()
        

    def compute_focal_loss(self, inputs, targets, gamma=2.0, input_sigmoid=False): 
        if input_sigmoid:
            p = inputs
        else:
            p = F.sigmoid(inputs)
        p = torch.clamp(p, 1e-7, 1-1e-7)
        p_t = p * targets + (1 - p) * (1 - targets)

        if input_sigmoid:
            ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        else:
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss

    def compute_dice_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            batch_mask, 
            input_sigmoid=False
        ):
        if not input_sigmoid: inputs = inputs.sigmoid()
        inputs = inputs * batch_mask
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.mean()

    def get_coarse_mask(self, fine_pcd, fine_offset, coarse_pcd, coarse_offset, **mask_dict):
        indexes, _ = glib.knn(coarse_pcd, fine_pcd, k=1, query_offset=coarse_offset, base_offset=fine_offset)
        indexes = indexes.flatten()
        result = {}
        for k, mask in mask_dict.items():
            result[k] = mask[indexes]
        return result


    def to_fine_map(self, fine_pcd, fine_offset, coarse_pcd, coarse_offset, **map_dict):
        indexes, _ = glib.knn(fine_pcd, coarse_pcd, k=1, query_offset=fine_offset, base_offset=coarse_offset)
        indexes = indexes.flatten()
        result = {}
        for k, v in map_dict.items():
            result[k] = v[indexes]
        return result

    def get_loss(self, logits, label_mask, offset, want_iou=True, input_sigmoid=False):
        batch_logits, batch_mask = glib.to_dense_batch(logits, offset, input_offset=True)
        batch_label_mask, _ = glib.to_dense_batch(label_mask, offset, input_offset=True)

        dice_loss = self.compute_dice_loss(batch_logits, batch_label_mask, batch_mask, input_sigmoid=input_sigmoid)
        focal_loss = self.compute_focal_loss(batch_logits, batch_label_mask.float(), input_sigmoid=input_sigmoid) # (B, L)
        focal_loss = focal_loss * batch_mask
        focal_loss = focal_loss.sum(dim=1) / batch_mask.sum(dim=1)

        metric_dict = {}
        if want_iou:
            metric_dict['iou'] = self.compute_iou(batch_logits, batch_label_mask, batch_mask)
        
        return {'focal_loss': focal_loss.mean(), 'dice_loss': dice_loss}, metric_dict
    
    

class RegionMatchingNetwork(nn.Module):
    
    def __init__(self, k=16, 
                in_dim=14, 
                hidden_dim=128, n_heads=4, matching_temperature=1.0, max_condition_num=-1,
                focal_gamma=2.0, #stage1_query_layers=DEFAULT_STAGE1_QUERY_LAYERS, 

                stage1_grid_sizes=(0.015, 0.03),
                stage1_depths=(2, 3, 2),
                stage1_dec_depths=(1, 1),
                stage1_hidden_dims=(256, 384), 
                
                stage2_layers=None, **kwargs):
        super().__init__()
        stage2_layers = stage2_layers or [
            "positioning(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "self(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "positioning(src,tgt)",   
            "cross(src,tgt,tgt,src)",
            "self(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "positioning(src,tgt):no_emb",
        ]
        match_block = glib.DualSoftmaxReposition(hidden_dim, matching_temperature, max_condition_num=max_condition_num, 
                                                focal_gamma=focal_gamma, one_way=True)
        self.in_dim = in_dim
        self.k = k
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.reposition = glib.DualSoftmaxReposition(hidden_dim, matching_temperature, max_condition_num=max_condition_num, 
                                                focal_gamma=focal_gamma, use_projection=False, one_way=True)
        
        # self.stage1 = glib.KnnTransformerNetwork(stage1_layers, glib.make_knn_transformer_layers(stage1_layers, hidden_dim, n_heads))
        self.stage1 = glib.PointTransformerNetwork(grid_sizes=stage1_grid_sizes, 
                                                depths=stage1_depths, 
                                                dec_depths=stage1_dec_depths, 
                                                hidden_dims=(hidden_dim,) + stage1_hidden_dims, 
                                                n_heads=(4, 8, 8), ks=(16, 24, 32), in_dim=in_dim)

        def make_position_layer(typ):
            if 'no_emb' in typ:
                return dc(match_block)
            else:
                return (dc(match_block), nn.Linear(3, hidden_dim))
            
        def clean_names(layers):
            return [l.split(":")[0] for l in layers]
        
        self.stage2 = glib.KnnTransformerNetwork(clean_names(stage2_layers), [
            make_position_layer(l) if 'positioning' in l else glib.make_knn_transformer_one_layer(l, hidden_dim, n_heads)
            for l in stage2_layers if "embedding" not in l    
        ])
        
        self.sequence_embed = nn.Linear(hidden_dim, hidden_dim)

    
    def forward(self, batch):
        meta_data = batch['meta']
        bsize = len(meta_data['ko_correspondence']) 
        dev = batch['src']['t']['pcd'].device
        output, loss_dict, metric_dict = {}, {}, {}

        src, tgt = frame_data = batch['src']['t'], batch['tgt']['t']
        
        with Section("Initial Data Preparation"):
            for item in [src, tgt]:
                item['offset'] = glib.batch2offset(item['batch_index'])
                item['robot_pcd'] = glib.batch_X_transform_flat(item['pcd'], item['batch_index'], item['X_to_robot_frame'])

                item['feat'] = torch.cat([item[k] for k in ['pcd', 'rgb', 'normal', 'robot_pcd']], dim=1)
                item['open'], item['ignore_col'] = item['open'].reshape(-1, 1).float(), item['ignore_col'].reshape(-1, 1).float()
                item['feat'] = torch.cat([item['feat'],
                                            glib.expand(item['open'], item['batch_index']),
                                            glib.expand(item['ignore_col'], item['batch_index'])], dim=1)
        
                # item['feat'] = self.input_embed(item['feat'])
                item['knn'] = glib.knn(item['pcd'], item['pcd'], self.k, query_offset=item['offset'])[0]
        
        with Section("Stage 1. base"):
            for item in [src, tgt]:
                # item['feat'] = self.stage1(feat={'p': item['feat']}, coord={'p': item['pcd']}, knn_indexes={'p2p': item['knn']})['p']
                item['feat'] = self.stage1([item['pcd'], item['feat'], item['offset']])[1]

        
        with Section("Creating key feature"):
            for name, mask in [('ko', src['key_mask']),]: #  ('ctx', ~src['key_mask'])
                src[name + '_batch_index'] = src['batch_index'][mask]
                src[name + '_offset'] = glib.batch2offset(src[name + '_batch_index'])
                src[name + '_pcd'] = src['pcd'][mask]
                src[name + '_feat'] = src['feat'][mask]

            src['ko2ko'], _ = glib.knn(src['ko_pcd'], src['ko_pcd'], self.k, query_offset=src['ko_offset'])

        with Section("add instruction (optional)"):
            for k, item in [('src', src), ('tgt', tgt)]:
                position_mask = item['noisy_position_mask' if self.training and ('noisy_position_mask' in item) else 'position_mask']
                seq_triangular = glib.distance_embed(position_mask[:, None], scale=1., num_pos_feats=self.hidden_dim)
                seq_embed = self.sequence_embed(seq_triangular[:, 0, :])
                seq_embed = (position_mask != -1)[:, None] * seq_embed
                if k == 'src':
                    item['ko_feat'] = item['ko_feat'] + seq_embed[item['key_mask']]
                else:
                    item['feat'] = item['feat'] + seq_embed
        
        src2tgt_kindexes, _ = glib.knn(src['ko_pcd'], tgt['pcd'], self.k, query_offset=src['ko_offset'], base_offset=tgt['offset'])
        tgt2src_kindexes, _ = glib.knn(tgt['pcd'], src['ko_pcd'], self.k, query_offset=tgt['offset'], base_offset=src['ko_offset'])

        with Section("Stage 2. registration"):
            coord, feat, knn_indexes, position_outputs = self.stage2(feat={'src': src['ko_feat'], 'tgt':tgt['feat']}, 
                                                coord={'src': src['ko_pcd'], 'tgt': tgt['pcd']}, 
                batch_index={'src': src['ko_batch_index'], 'tgt': tgt['batch_index']},
                knn_indexes={'src2src': src['ko2ko'], 'tgt2tgt': tgt['knn'],
                            'src2tgt': src2tgt_kindexes, 'tgt2src': tgt2src_kindexes})
        
        src_tp1_position = src['robot_position_t+1']
        conf_matrix = position_outputs[-1]['conf_matrix'].detach()
        R0, t0, cond = self.reposition.arun(conf_matrix, src['ko_pcd'], src['ko_batch_index'], tgt['pcd'], tgt['batch_index'])            
        tgt_tp1_position_hat = glib.batch_Rt_transform(src_tp1_position, R0, t0)
        output['Rt'] = [R0, t0]
        output['predict_frame'] = tgt_tp1_position_hat
        
        if self.training:
            conf_matrix = position_outputs[-1]['conf_matrix']
            correspondence = [] 
            for m1, m2 in zip(meta_data['correspondence'], meta_data['ko_correspondence']):
                correspondence.append(torch.cat([m2[:, 0].reshape(-1, 1), m1[:, 1].reshape(-1, 1)], dim=1))

            gt_matrix = self.reposition.to_gt_correspondence_matrix(conf_matrix, correspondence)
            for i, out in enumerate(position_outputs):
                corr_loss = self.reposition.compute_matching_loss(out['conf_matrix'], gt_matrix=gt_matrix)
                loss_dict[f'position_corr_loss_{i}'] = corr_loss
        
        if tgt.get('robot_position_t+1', None) is not None:
            reg_action_l1dist = torch.abs(tgt_tp1_position_hat - tgt['robot_position_t+1']).sum(dim=-1)
            metric_dict['action(reg)_l1_t'] = reg_action_l1dist[:, 0].mean()
            metric_dict['action(reg)_l1_xyz'] = reg_action_l1dist[:, 1:].mean()
            
        output['conf_matrix'] = conf_matrix
        return {'output': output, 
                'loss_dict': loss_dict, 'metric_dict': metric_dict}
        
        

class RegionMatchingNetwork_fine(nn.Module):
    def __init__(self, k=16, 
                in_dim=14, 
                hidden_dim=128, n_heads=4, matching_temperature=1.0, max_condition_num=-1,
                focal_gamma=2.0, 

                stage1_grid_sizes=(0.015, 0.03),
                stage1_depths=(2, 3, 2),
                stage1_dec_depths=(1, 1),
                stage1_hidden_dims=(256, 384), 
                
                stage2_layers=None):
        super().__init__()
        stage2_layers = [
            "positioning(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "self(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "positioning(src,tgt)",   
            "cross(src,tgt,tgt,src)",
            "self(src,tgt)",
            "cross(src,tgt,tgt,src)",
            "positioning(src,tgt):no_emb",
        ]

        match_block = glib.DualSoftmaxReposition(hidden_dim, matching_temperature, max_condition_num=max_condition_num, 
                                                focal_gamma=focal_gamma)
        self.in_dim = in_dim
        self.k = k
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.reposition = glib.DualSoftmaxReposition(hidden_dim, matching_temperature, max_condition_num=max_condition_num, 
                                                focal_gamma=focal_gamma, use_projection=False)
        
        self.stage1 = glib.PointTransformerNetwork(grid_sizes=stage1_grid_sizes, 
                                                depths=stage1_depths, 
                                                dec_depths=stage1_dec_depths, 
                                                hidden_dims=(hidden_dim,) + stage1_hidden_dims, 
                                                n_heads=(4, 8, 8), ks=(16, 24, 32), in_dim=in_dim)

        def make_position_layer(typ):
            if 'no_emb' in typ:
                return dc(match_block)
            else:
                return (dc(match_block), nn.Linear(3, hidden_dim))
            
        def clean_names(layers):
            return [l.split(":")[0] for l in layers]
        
        self.stage2 = glib.KnnTransformerNetwork(clean_names(stage2_layers), [
            make_position_layer(l) if 'positioning' in l else glib.make_knn_transformer_one_layer(l, hidden_dim, n_heads)
            for l in stage2_layers if "embedding" not in l    
        ])
    
    def forward(self, batch):
        meta_data = batch['meta']
        bsize = len(meta_data['ko_correspondence']) 
        dev = batch['src']['t']['pcd'].device
        output, loss_dict, metric_dict = {}, {}, {}
        src, tgt = frame_data = batch['src']['t'], batch['tgt']['t']

        for item in [src, tgt]:
            item['offset'] = glib.batch2offset(item['batch_index'])
            item['robot_pcd'] = glib.batch_X_transform_flat(item['pcd'], item['batch_index'], item['X_to_robot_frame'])

            item['feat'] = torch.cat([item[k] for k in ['pcd', 'rgb', 'normal', 'robot_pcd']], dim=1)
            item['open'], item['ignore_col'] = item['open'].reshape(-1, 1).float(), item['ignore_col'].reshape(-1, 1).float()
            item['feat'] = torch.cat([item['feat'],
                                        glib.expand(item['open'], item['batch_index']),
                                        glib.expand(item['ignore_col'], item['batch_index'])], dim=1)
        
        for item in [src, tgt]:
            item['feat'] = self.stage1([item['pcd'], item['feat'], item['offset']])[1]
        
        for item in [src, tgt]:
            item['key_batch_index'] = item['batch_index'][item['key_mask']]
            item['key_offset'] = glib.batch2offset(item['key_batch_index'])
            item['key_pcd'] = item['pcd'][item['key_mask']]
            item['key_pcd(origin)'] = item['key_pcd'].clone()
            item['key_feat'] = item['feat'][item['key_mask']]

            item['key_center'] = global_mean_pool(item['key_pcd'], item['key_batch_index'])
            item['key_pcd'] -= glib.expand(item['key_center'], item['key_batch_index'])

            item['robot_position(origin)'] = item['robot_position'].clone()
            item['robot_position'] -= item['key_center'][:, None, :]
            if 'robot_position_t+1' in item:
                item['robot_position_t+1(origin)'] = item['robot_position_t+1'].clone()
                item['robot_position_t+1'] -= item['key_center'][:, None, :]


        knn_indexes = dict(src2tgt=glib.knn(src['key_pcd'], tgt['key_pcd'], self.k, query_offset=src['key_offset'], base_offset=tgt['key_offset'])[0], 
                tgt2src=glib.knn(tgt['key_pcd'], src['key_pcd'], self.k, query_offset=tgt['key_offset'], base_offset=src['key_offset'])[0], 
                src2src=glib.knn(src['key_pcd'], src['key_pcd'], self.k, query_offset=src['key_offset'])[0],
                tgt2tgt=glib.knn(tgt['key_pcd'], tgt['key_pcd'], self.k, query_offset=tgt['key_offset'])[0])
        

        coord, feat, knn_indexes, position_outputs = self.stage2(feat={'src': src['key_feat'], 'tgt':tgt['key_feat']}, 
                                                coord={'src': src['key_pcd'], 'tgt': tgt['key_pcd']}, 
                batch_index={'src': src['key_batch_index'], 'tgt': tgt['key_batch_index']},
                knn_indexes=knn_indexes)
        
        src_tp1_position = src['robot_position_t+1']
        conf_matrix = position_outputs[-1]['conf_matrix'].detach()
        R0, t0, cond = self.reposition.arun(conf_matrix, src['key_pcd'], src['key_batch_index'], tgt['key_pcd'], tgt['key_batch_index'])            
        tgt_tp1_position_hat = glib.batch_Rt_transform(src_tp1_position, R0, t0)
        output['transformation'] = self.reposition.arun(conf_matrix, src['key_pcd(origin)'], src['key_batch_index'], 
                                                        tgt['key_pcd(origin)'], tgt['key_batch_index'])        
        tgt_tp1_position_hat_origin = glib.batch_Rt_transform(src['robot_position_t+1(origin)'], *output['transformation'][:2])
        output['predict_frame'] = tgt_tp1_position_hat_origin
        output['conf_matrix'] = position_outputs[-1]['conf_matrix']
        if self.training:
            conf_matrix = position_outputs[-1]['conf_matrix']
            gt_matrix = self.reposition.to_gt_correspondence_matrix(conf_matrix, meta_data['ko_correspondence'])
            for i, out in enumerate(position_outputs):
                corr_loss = self.reposition.compute_matching_loss(out['conf_matrix'], gt_matrix=gt_matrix)
                loss_dict[f'position_corr_loss_{i}'] = corr_loss
        
        if tgt.get('robot_position_t+1', None) is not None:
            reg_action_l1dist = torch.abs(tgt_tp1_position_hat - tgt['robot_position_t+1']).sum(dim=-1)
            metric_dict['action(reg)_l1_t'] = reg_action_l1dist[:, 0].mean()
            metric_dict['action(reg)_l1_xyz'] = reg_action_l1dist[:, 1:].mean()
            
        return {'output': output, 
                'loss_dict': loss_dict, 'metric_dict': metric_dict}