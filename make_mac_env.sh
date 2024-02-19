conda create -n ml_workshop_mac python=3.11
conda activate ml_workshop_mac
pip install jaxlib jax equinox optax diffrax tqdm matplotlib tensorflow einops ipywidgets
python setup.py
conda env export --no-builds > env_mac_nb.yml
conda env export > env_mac.ml
conda deactivate
