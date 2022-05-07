import os
from copy import deepcopy
import logging
from collections import OrderedDict

import torch
from torch import nn


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def save_model(model,
               optimizer,
               epoch,
               save_path,
               model_ema=None):
    save_state = {
        "epoch": epoch,
        "state_dict": get_state_dict(model),
        "optimizer": optimizer.state_dict()
    }
    if model_ema is not None:
        save_state['state_dict_ema'] = get_state_dict(model_ema)
    save_path = os.path.join(save_path, f"epoch {epoch}.pth")
    torch.save(save_state, save_path)


def load_state_dict(ckpt_path, use_ema=True):
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu') # load to cpu to avoid memory leak
        state_dict_key = ""
        if isinstance(ckpt, dict):
            if use_ema and ckpt.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and ckpt.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in ckpt:
                state_dict_key = 'state_dict'
            elif 'model' in ckpt:
                state_dict_key = 'model'
        state_dict = clean_state_dict(ckpt[state_dict_key] if state_dict_key else ckpt)
        logging.info("Loaded {} from checkpoint '{}'".format(state_dict_key, ckpt))
        return state_dict
    else:
        logging.error("No checkpoint found at '{}'".format(ckpt_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=True, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


class ModelEma(nn.Module):
    """ This class perform model weight EMA
    model_weights = decay * model_weights + (1 - decay) * new_model_weights
    """
    def __init__(self, model, decay=0.9, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
