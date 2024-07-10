from os import path
import datetime

import random
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed
from torchvision import transforms


from model.model import S2MModel
from dataset.lvis_dataset import  InteractiveSegmentationDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.empatches import EMPatches



"""
Initial setup
"""


num_classes = 3  # Set the number of classes


torch.backends.cudnn.benchmark = True

# Init distributed environment
distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print('I am rank %d in the world of %d!' % (local_rank, world_size))

"""
Model related
"""
if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))

    # Construct rank 0 model
    model = S2MModel(para,num_classes=num_classes, logger=logger, 
                    save_path=path.join('saves', long_id, long_id) if long_id is not None else None, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct models of other ranks
    model = S2MModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
else:
    total_iter = 0
    
if para['load_deeplab'] is not None:
    model.load_deeplab(para['load_deeplab'])

if para['load_network'] is not None:
    model.load_network(para['load_network'])

"""
Dataloader related
"""
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(train_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank, shuffle=False)
    train_loader = DataLoader(train_dataset, para['batch_size'], sampler=train_sampler, num_workers=8,
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return train_sampler, train_loader

"""
Dataset related
"""

# Example usage
data_dir = "/home/spoch/Documents/private/scribble_gen/patched"

#histoire de chemin d'image quine colle pas. il faut que les it des path soient avant le msk ou scr

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
dataset = InteractiveSegmentationDataset(data_dir, num_classes, transform=transform)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)




total_epoch = math.ceil(para['iterations']/len(train_loader))
print('Number of training epochs (the last epoch might not complete): ', total_epoch)

# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    for e in range(total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        
        # Crucial for randomness! 
        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1
            if total_iter >= para['iterations']:
                break
finally:
    if not para['debug'] and model.logger is not None and total_iter>1000:
        model.save(total_iter)
    # Clean up
    distributed.destroy_process_group()
