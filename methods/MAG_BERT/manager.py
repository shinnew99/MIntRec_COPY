import torch
import torch.nn.functional as F
import logging
from torch import nn, optim
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup

__all__ = ['MAG_BERT']


class MAG_BERT:
    def __init__(self, args, data, model):
        self.logger = logging.getLogger(args.logger_name)
        