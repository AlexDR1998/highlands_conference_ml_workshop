import jax
import jax.numpy as np
import jax.random as jr
import optax
import einops
import equinox as eqx 
import diffrax as dx
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()