# Diffusion Exploration

Diffusion policy taken from [here](https://diffusion-policy.cs.columbia.edu/). 
* Vision ipynb based on [this](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing)


## Setup (Linux)
1. Create your virtual environment and install dependencies
```bash
python3.10 -m venv venv-diff
source venv-diff/bin/activate
pip install -r requirements.txt
```

2. If running the jupiter notebook, in VSCode, in the ipynb select venv-dex as your kernel.


## Running 
* Jupiter notebook (diffusion_policy_state_pusht_demo.ipynb): Run in VScode
* Training your own data:
```bash
python3 model_pipeline/collect_data.py
```

## Results
Trained on provided data:

<img src="./T_push_trained.gif" width="200">