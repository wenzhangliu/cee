# Reducing Action Space for Deep Reinforcement Learning via Causal Effect Estimation


## Requirements
We test our method with a gpu that can run CUDA 12.0. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with:
```
source activate cee
```
## Instructions 
### Pre-training: Training N-value network
First, we need to train an N-value network. For example, in the Unlock Pickup environment, run:
```
cd cee
python min_red/grid_search_minigrid.py
```

### Phase 2: Conduct task training
When the pre-training is complete, add the model to ***makppo/train.py/Env_mask_dict***```
```
python maskppo/grid_search_minigrid.py
```

Both of two phases  will produce 'log' folder, where all the outputs are going to be stored. The data and lines can be observed in tensorboard.


```
tensorboard --logdir log
```
Besides, Operation of the PurePPO algorithm:

```
python pureppo/grid_search_minigrid.py
```