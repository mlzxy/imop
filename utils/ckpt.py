import torch
import torch.nn as nn

def remove_dict_prefix(state_dict, prefix="module."):
    result = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        result[k] = v 
    return result

    
def get_model(mod):
    if hasattr(mod, 'module'):
        return mod.module
    else:
        return  mod
    
    
def freeze_model(mod):
    mod.eval()
    for p in mod.parameters():
        p.requires_grad = False
    
    
def compute_grad_norm(model):
    if isinstance(model, nn.Module):
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    else:
        grads = [param.grad.detach().flatten() for param in model if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm