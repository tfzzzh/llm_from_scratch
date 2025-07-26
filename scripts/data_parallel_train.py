import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from llm.config import read_config
from llm.trainer import DPTrainer
CONFIG_PATH = 'configs/dp_config.yaml'

def setup(rank, world_size, master_addr, master_port, backend, omp_num_threads):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["OMP_NUM_THREADS"] = omp_num_threads
    # multiple worker processes
    # "nccl" backend will use the NVIDIA NCCL collective communications library
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def main(rank, world_size):
    config = read_config(CONFIG_PATH)
    
    setup(
        rank,
        world_size,
        config['distributed']['MASTER_ADDR'],
        config['distributed']['MASTER_PORT'],
        config['distributed']['backend'],
        config['distributed']['omp_num_threads']
    )

    # set seed (make data loader sampling different data)
    seed = config['seed']
    torch.manual_seed(seed + 100 * rank)
    np.random.seed(seed + 100 * rank)

    print(f"processor with rank {rank} start")
    
    trainer = DPTrainer(config)
    trainer.train()

if __name__ == '__main__':
    config = read_config(CONFIG_PATH)
    world_size = config['distributed']['world_size']
    mp.spawn(fn=main, args=(world_size, ), nprocs=world_size, join=True)