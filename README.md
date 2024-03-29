# Highlands conference machine learning workshop

Included here are the instructions for setting up your python environment, and the jupyter notebooks to accompany the machine learning workshop in the PhD physics winter conference 2024

## Installation instructions
Follow these instructions to set up a conda environment with everyting installed, and then to download the dataset we'll be playing with. These instructions vary a little depending on your operating system:

### Windows
```
git clone https://github.com/AlexDR1998/highlands_conference_ml_workshop.git
cd highlands_conference_ml_workshop
conda env create -f env_windows.yml
conda activate ml_workshop
pip install notebook
python setup.py
```


### Ubuntu
```
git clone https://github.com/AlexDR1998/highlands_conference_ml_workshop.git
cd highlands_conference_ml_workshop
conda env create -f env_linux.yml
conda activate ml_workshop
pip install notebook
python setup.py
```


### mac OS
```
git clone https://github.com/AlexDR1998/highlands_conference_ml_workshop.git
cd highlands_conference_ml_workshop
conda env create -f env_mac.yml
conda activate ml_workshop
pip install notebook
python setup.py
```

### These didn't work!
If the conda environment didn't work, for whatever reason, just manually install the python libraries used in `setup.py`. Note that you need to be in python vesrsion 3.11 :
#### On windows:
```
pip install jax[cpu] optax equinox diffrax tensorflow matplotlib tqdm ipywidgets einops notebook
```

#### On macOS or Ubuntu:

```
pip install jaxlib jax[cpu] optax equinox diffrax tensorflow matplotlib tqdm ipywidgets einops notebook
```

If you have a gpu, feel free to change the `jax[cpu]` based on the instructions found here: https://jax.readthedocs.io/en/latest/installation.html
