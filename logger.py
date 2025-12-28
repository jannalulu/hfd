from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

import typing

class Logger():
    log_level = 0

def log_always(*args):
    rank_zero_info(args)

def log(*args):
    if Logger.log_level > 0:
        rank_zero_info(args)

# @rank_zero_only
# def print0(*args, 
#     sep: str | None = " ",
#     end: str | None = "\n",
#     file = None,
#     flush: typing.Literal[False] = False,
# ):
#     print(*args, sep=sep, end=end, file=file, flush=flush)

#import deepspeed
from torch.distributed import get_rank
#import os
def print0(*args, **kwargs):
    try:
        rank = get_rank()
    except:
        rank = -1
    if rank <= 0:
        print(*args, **kwargs)
