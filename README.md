# Distributional Reinforcement Learning using Feedback from External Knowledge Sources 

This repository contains the code for the paper:  Distributional Reinforcement Learning using Feedback from External Knowledge Sources. Follow the instructions here to install and run the code. 


## Atari Environments

See the algos\_discrete folder for the algorithmic implementations. 

### Installation

1. Ensure you have Python 3.8 to 3.11 installed. It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
python -m venv c51ef_env
```

```bash
c51ef_env\Scripts\activate
```

```bash
source c51ef_env/bin/activate 
```

2. Install Dependencies 

You will need gymnasium with the Atari suite, torch for the neural networks, and the ALE (Arcade Learning Environment) ROMs.

```bash
# Update pip
pip install --upgrade pip

# Install Gymnasium and Atari dependencies
pip install "gymnasium[atari, accept-rom-license]"

# Install PyTorch (Standard CPU version; use 'torch torchvision torchaudio' for GPU)
pip install torch numpy

# Install ALE-py to manage Atari ROMs
pip install ale-py

# Installation of Data Processing Tools (to handle the large Atari-Head datasets efficiently)

pip install h5py opencv-python

```


3. Hardward Requirements: 

Memory: At least 16GB RAM (The replay buffer stores $5 \times 10^5$ frames, which is memory-intensive).

Storage: Ensure you have enough space for the Atari-Head datasets if you are loading external human demonstration files.


### Data and Data Installation 

The Atari-Head dataset can be downloaded here: https://zenodo.org/records/10966777. The download contains a [README](./data/bridge_dataset/README.md) with more information about the format of the data. Follow all the installation instructions for Atari-Head (through the usage of Docker) in the [free-lunch-saliency](https://github.com/dniku/free-lunch-saliency) to install the requirements for using Atari-Head for training. 



### RL training

The script uses the v5 environment suffix to support the 30% sticky action requirement via the repeat\_action\_probability=0.3 argument. All the code for the Atari training can be found in the algos\_discrete folder. 

```bash
# To run Pong with 30% sticky actions
python c51_ef_scenario1.py ALE/Pong-v5 Pong_predictor_85.pth

# To run Asterix with 30% sticky actions
python c51_ef_scenario1.py ALE/Asterix-v5 Asterix_predictor_85.pth

```

Similarly modify the command to run training on the corresponding Atari game. To run this, you need to provide both the environment and the path to your pre-trained weight file. All the baseline implementations are for Scenario 1. For all the Scenario 2 experiments you will need to implement the noisy reward function in the algorithms (see c51\_ef\_scenario2.py for an example). For the additional experiments in Subsection 6.3 -- 6.6 of the paper, run the corresponding files in the _additionalexperiments_ folder. 



### Implementation Notes 


External Predictor: The predictor model in the code is currently initialized with random weights. For a true replication of Scenario 1, you should pre-train this model on the Atari-Head dataset, then load those weights using predictor.load\_state\_dict(torch.load('path\_to\_weights.pth')).

You can use the pretraining.py script to build the predictor. This script assumes you have converted the Atari-Head .txt or .json logs into a structured format (like a .npz or h5py file) containing pairs of frames and actions.



Observation Space: The script expects 4-frame stacked grayscale images (standard for Atari). Ensure your environment wrapper handles the AtariPreprocessing if the raw gym.make does not suffice for your specific preprocessing pipeline.




## SMARTS Environments

See the algos\_continuous folder for the algorithmic implementations. 

### Installation

1. SMARTS requires specific system dependencies (specifically sumo and spatialindex).

```bash
sudo apt-get install -y libspatialindex-dev sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
```


2. Environment Setup 

Follow the instructions provided in [SMARTS](https://smarts.readthedocs.io/en/latest/setup.html). 


```bash
# For Mac OS X users, ensure XQuartz is pre-installed.
# Install the system requirements. You may use the `-y` option to enable
# automatic assumption of "yes" to all prompts to avoid timeout from
# waiting for user input.
$ bash utils/setup/install_deps.sh

# This should install the latest version of SMARTS from package index (generally PyPI).
$ pip install 'smarts[camera-obs,sumo,examples]'
```



### Data and Data Installation 

Follow the instructions in this [link](https://smarts.readthedocs.io/en/latest/ecosystem/waymo.html) to download the integrate SMARTS with the Waymo dataset and build the predictor model. 


### RL training


```bash
# Run with the path to your pre-trained expert model (Scenario 1 proxy)
python c51_ef_scenario1.py --scenario scenarios/sumo/intersections/4lane --expert predictor_model.pth
```

Similarly modify the command to train other algorithms from the folder. 
