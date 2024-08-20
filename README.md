# Scribble based segmentation

so far there is a border + interlan multiclass scribble generator, It takes the original mask of N classes and returns scribbles as a new image array with the values repreenting the same classes as the riginal mask.

There is also an iterable datalodar whhich idea is to create several samples from one single image/mask pair. It will create a subset of classes and trat it as a single sample, generating scribbles for the sublcasses seected. It is important that always there is the case in which a sample represents the whole image withut scribbles so the model learns to segment with no prompts.

## How to use 

Clone this repository : 


### How to use the scribble generator ?

Create and activate a new environment and install the packages in `requirements.txt`  **TODO**

Execute the following command : 

```python gen_scribbles.py --imgs path-to-imgs --masks path-to-masks```

You can change few parameters as described below : 

```
gen_scribbles.py [-h] [--imgs IMGS] [--masks MASKS] [--save SAVE]
                        [--patch PATCH] [--config CONFIG]
                        [--patch-size PATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --imgs IMGS           Path of the image file or directory
  --masks MASKS         Path of the masks file or directory
  --save SAVE           Path for saving scribbles
  --patch PATCH         Path for saving patched images
  --config CONFIG       Path of the configuration file
  --patch-size PATCH_SIZE
                        Size in pixels for the square patches
```

### How to train the model ?

Execute : 

```torchrun scr2msk/train.py --id your-xp-id```

Here also you can change some parameters : 

```
usage: train.py [-h] [--source SOURCE] [-i ITERATIONS] [--lr LR]
                [--steps [STEPS [STEPS ...]]] [--classes CLASSES]
                [-b BATCH_SIZE] [--gamma GAMMA] [--load_network LOAD_NETWORK]
                [--load_deeplab LOAD_DEEPLAB] [--load_model LOAD_MODEL]
                [--id ID] [--debug] [--local_rank LOCAL_RANK]

optional arguments:
  -h, --help            show this help message and exit
  --source SOURCE       Path of the dataset repository
  -i ITERATIONS, --iterations ITERATIONS
                        Total number of iterations
  --lr LR               Learning rate
  --steps [STEPS [STEPS ...]]
                        Step at which the learning rate decays
  --classes CLASSES     Number of classes to give to the model
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  --gamma GAMMA         Gamma used in learning rate decay
  --load_network LOAD_NETWORK
                        Path to pretrained network weight only
  --load_deeplab LOAD_DEEPLAB
  --load_model LOAD_MODEL
                        Path to the model file, including network, optimizer
                        and such
  --id ID               Experiment UNIQUE id, use NULL to disable logging to
                        tensorboard
  --debug               Debug mode which logs information more often
  --local_rank LOCAL_RANK
                        Local rank of this process
```


## Questions:
  - are the generateds scribbles good enough?
  - Is the idea of the dataloader encesary at all? or single image/mask is enough



#TODO: 
  - use the scribble gnerator and create a dataset based on some segmentation datasets (COCO, Cityscapes, anyone)
  - train a DeepLab model in which the scribbles are just other channel https://github.com/hkchengrex/Scribble-to-Mask
  - experiment with more architectures
  - fine tune in specific datasetwith few samples (clouds)
  - train just with small dataset (clouds) 
