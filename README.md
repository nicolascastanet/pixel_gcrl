### Installation

- Create venv and install requirements
```
    python3 -m venv my_python_env
    source ~/my_python_env/bin/activate
    pip install --upgrade pip
    pip3 install -r requirements.txt
```

- Install a specific version of torch
```
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --trusted-host download.pytorch.org
```

Fork this repository with the following path:  `~/Git/pixel_gcrl`


### Organization

The `xpag` directory contains blocks for the RL algorithm, buffers, setter, HER, learning/plotting functions etc... See readme file inside the directory for more informations. 
The main RL training loop is located in the file `xpag/tools/learn.py`, other training such as VAE for pixel based input are done in this file.

The `train.py` script is used to launch experiments, `make_setters.py` is used to build the blocks of every setters methods such as `SVGG`, `MEGA`, `Goal-GAN` or `RIG`. The setters implementation are located in `xpag/setters/setter.py`, for `SVGG`, the folder `xpag/svgg/` contains many building block of the methods such its energy based distribution criterion in `xpag/svgg/criterion.py`, or the algoritmh `SVGD` in `xpag/svgg/svgd.py`.


### launch Experiments

- launch SVGG with TQC algorithm on a Maze with coordinates observations (x,y):
    ```
    python train.py rl_algo=TQC env.maze_type=square_pbcs_0 method=svgg actor.evaluate_every_x_steps=10000 method.name=SVGG_TQC/MAZE_0 method.plot=True
    ```

- launch RIG with TQC algorithm on a Maze with pixel based observation (top-down view):

    ```
    python train.py rl_algo=TQC env.maze_type=square_pbcs_0 method=rig env.from_pixel=True env.latent_dim_obs=2 actor.evaluate_every_x_steps=10000 method.name=RIG_TQC/MAZE_0 method.plot=True
    ```

