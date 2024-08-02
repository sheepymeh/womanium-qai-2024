from PIL import Image
import jax
import jax.numpy as jnp
import os
from tqdm import tqdm

files = [os.path.join(folder, f) for folder in os.listdir('data/train') if os.path.isdir(f'data/train/{folder}') for f in os.listdir(f'data/train/{folder}')]
running_mean, running_var = 0., 0.

for f in tqdm(files):
	image = Image.open('data/train/' + f)
	if image.size != (800, 974):
		print(f)

	image = jnp.array(image) / 255
	running_mean += image.mean()

running_mean /= len(files)
print('mean:', running_mean)

for f in tqdm(files):
	image = Image.open('data/train/' + f)
	image = jnp.array(image) / 255
	running_var += ((image - running_mean) ** 2).mean()

running_var /= len(files)
print('var:', running_var)