import torch
import multiprocessing
import traceback
from termcolor import colored
from tqdm import tqdm
import os.path as osp
from copy import copy
from omegaconf import OmegaConf
from utils import configurable, DictConfig, config_to_dict
from utils.structure import load_pkl
import torch.multiprocessing as mp
from utils.dist import find_free_port
from utils.ckpt import remove_dict_prefix, compute_grad_norm
from time import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optim import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel
from runstats import Statistics
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
from utils.object import flat2d, to_item, color_terms, detach

from data import RLBenchDataset, RLBenchTransitionPairDataset, RLBenchCollator, to_device
from network import InvariantRegionNetwork, RegionMatchingNetwork, RegionMatchingNetwork_fine


def main_single(rank: int, cfg: DictConfig, port: int, log_dir:str):
    world_size = cfg.train.num_gpus
    if world_size == 0: world_size = 1
    ddp, on_master = world_size > 1, rank == 0
    print(f'Rank - {rank}, master = {on_master}')
    if ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = rank 
    if cfg.train.num_gpus == 0: device = 'cpu'
    else: 
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    if on_master:
        logfile = open(osp.join(log_dir, 'log.txt'), "w")

    def log(msg, printer=print):
        if on_master:
            print(msg, file=logfile, flush=False)
            printer(msg)
        
    log(f"「 {cfg.notes} 」")
        
    if cfg.train.tensorboard and rank == 0: 
        writer = SummaryWriter(log_dir=osp.join(log_dir, 'tensorboard'), max_queue=10000)
    if cfg.train.wandb and rank == 0:
        import wandb
        wandb.init(project="imop", config=config_to_dict(cfg), 
                notes=cfg.notes,
                name=f'{osp.basename(osp.dirname(log_dir))}_{osp.basename(log_dir)}')
    
    def log_metrics(stats):
        if cfg.train.tensorboard and rank == 0: 
            for k, v in stats.items(): writer.add_scalar(k, v, i)
        if cfg.train.wandb and rank == 0:  
            wandb.log(stats)
    
    lr = cfg.train.lr * (world_size * cfg.train.bs)
    cos_dec_max_step = cfg.train.epochs * cfg.train.num_transitions_per_epoch // cfg.train.bs
    log(f'cosine learning rate - max steps {cos_dec_max_step}')

    collate_fn = RLBenchCollator(use_segmap=False)
    model_kwargs = config_to_dict(cfg.model) 
    model_type = model_kwargs.pop('type')

    if model_type == 'invariant_region':
        model = InvariantRegionNetwork(**model_kwargs) 
    elif model_type == 'region_match': 
        model = RegionMatchingNetwork(**model_kwargs)
    elif model_type == 'region_match_fine': 
        model = RegionMatchingNetwork_fine(**model_kwargs)
    else:
        raise KeyError(model_type)

    if cfg.train.checkpoint: 
        log(f"loading checkpoint from {cfg.train.checkpoint}")
        load_result = model.load_state_dict(remove_dict_prefix(torch.load(cfg.train.checkpoint), prefix="module."), strict=False)
        log(f"load result: {load_result}")

    model = model.train().to(device)   
    if ddp:
        model = DistributedDataParallel(model, device_ids=[device])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    after_scheduler = CosineAnnealingLR(optimizer, T_max=cos_dec_max_step, eta_min=lr / 100)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.train.warmup_steps, after_scheduler=after_scheduler)

    start_step = 0
    total_sample_num = cfg.train.num_transitions_per_epoch * cfg.train.epochs
    total_sample_num -= (start_step * world_size * cfg.train.bs)

    db = RLBenchDataset(grid_size=cfg.data.grid_size, cache_to=cfg.data.db_cache, path=cfg.data.db_path, cache_mode='read',
                        color_only_instructions=cfg.data.color_only_instructions,
                        min_max_pts_per_obj=getattr(cfg.data, 'max_pts', 5000)) # just use the default parameters

    assert osp.exists(cfg.data.pairs_cache)
    pair_db = RLBenchTransitionPairDataset(db, cache_to=cfg.data.pairs_cache, size=total_sample_num, use_aug=cfg.data.aug, 
                                        correspondence=cfg.data.correspondence,
                                        align_twice=cfg.data.align_twice, include_T=cfg.data.include_T, noisy_mask=cfg.data.noisy_mask)
    dataloader = DataLoader(pair_db,
        batch_size=cfg.train.bs,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        collate_fn=collate_fn,
        drop_last=False)

    start = time()
    run_stats = {}
    total_steps = len(dataloader)

    for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
        batch = {k: to_device(v, device) for k,v in batch.items()}
        result = model(batch)
        loss_dict = result['loss_dict']
        
        optimizer.zero_grad(set_to_none=True)
        if 'total' not in loss_dict: 
            if hasattr(cfg.train, 'loss_weight'):
                loss_dict['total'] = 0
                for k, v in loss_dict.items():
                    weight = 1.0
                    for loss_name, loss_weight in config_to_dict(cfg.train.loss_weight).items():
                        if loss_name in k: 
                            weight = loss_weight 
                            break
                    loss_dict['total'] += (v * weight)
            else:
                loss_dict['total'] = sum(loss_dict.values())
        loss_dict['total'].backward()

        overall_grad_norm = compute_grad_norm(model)
        if cfg.train.grad_clip_after >= 0 and i >= cfg.train.grad_clip_after:
            if i == cfg.train.grad_clip_after: log("Start gradient clipping")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_value.overall)

        optimizer.step()
        scheduler.step()

        
        if rank == 0:
            loss_dict = {**loss_dict, **result.get('metric_dict', {})}

            for k, v in loss_dict.items():
                if k not in run_stats:
                    run_stats[k] = Statistics()
                    
            stat_dict = copy(loss_dict)
            for k in run_stats:
                if k in loss_dict:
                    run_stats[k].push(detach(loss_dict[k]))
                stat_dict[k] = run_stats[k].mean()
            
            loss_dict['lr'] = stat_dict['lr'] = scheduler.get_last_lr()[0]
            loss_dict['grad_norm'] = stat_dict['grad_norm'] = overall_grad_norm.detach()

            log_metrics(loss_dict)

            if i % cfg.train.log_freq == 0:
                msg = f"[step:{str(i + start_step).zfill(8)} time:{time()-start:.01f}s] " + " ".join([f"{k}:{to_item(v):.04f}" for k, v in sorted(stat_dict.items())])
                log(msg, printer=tqdm.write)
            if i != 0 and (i % cfg.train.save_freq == 0 or i == total_steps - 1):
                log(f"checkpoint to {log_dir} at step {i + start_step} and reset running metrics", printer=tqdm.write)
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler, 'step': i}, f'{log_dir}/{str(i + start_step).zfill(8)}.pth')
                run_stats = {}


@configurable()
def main(cfg: DictConfig):
    if cfg.train.num_gpus <= 1:
        main_single(0, cfg, -1, cfg.output_dir)
    else:
        port = find_free_port()
        mp.spawn(main_single, args=(cfg, port, cfg.output_dir), nprocs=cfg.train.num_gpus, join=True)


if __name__ == "__main__":
    main()