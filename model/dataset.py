import os
import json
import jax
import jax.numpy as jnp
import dm_pix as pix

import numpy as np
from PIL import Image
from grain.python import RandomAccessDataSource, Sampler, RecordMetadata, RandomMapTransform

mean = jnp.array([0.24085431])
var = jnp.array([0.01992414])

@jax.jit
def train_transform(images: jax.Array, key: jax.Array) -> jax.Array:
	n, h, w, c = images.shape

	images = jax.image.resize(images, (n, 512, 512, c), method='bicubic')
	images = pix.random_flip_left_right(key, images)
	images = pix.random_flip_up_down(key, images)
	images = pix.random_crop(key, images, (n, 224, 224, c))
	images = images / 255
	images = jax.nn.standardize(images, mean=mean, variance=var, axis=(2, 3))

	return images

@jax.jit
def val_transform(images: jax.Array, key: jax.Array) -> jax.Array:
	n, h, w, c = images.shape

	images = jax.image.resize(images, (n, 224, 224, c), method='bicubic')
	images = images / 255
	images = jax.nn.standardize(images, mean=mean, variance=var)

	return images


class ImageDataSource(RandomAccessDataSource[tuple[Image.Image, int]]):
	def __init__(self, path, split, num_classes = 6):
		self.image_dir = os.path.join(path, split)
		with open(os.path.join(self.image_dir, f'{split}.json')) as f:
			data = json.load(f)
			self.images = tuple(data.keys())
			self.labels = np.array(tuple(data.values()))
		self.num_classes = num_classes

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(self, idx) -> tuple[Image.Image, int]:
		image_path = os.path.join(self.image_dir, self.images[idx])
		image = Image.open(image_path).convert('RGB')
		label = self.labels[idx].item()
		return image, label


class WeightedIndexSampler(Sampler):
	def __init__(self, weights: np.ndarray, seed: int, num_epochs: int = 1):
		assert num_epochs > 0
		self._num_records = len(weights)
		self._max_index = self._num_records * num_epochs
		self._weights = weights
		self._seed = seed
		self._rng = np.random.Generator(np.random.Philox(self._seed))
		self._record_keys = self._rng.choice(self._num_records, size=self._max_index, replace=True, p=self._weights)

	def __getitem__(self, index: int) -> RecordMetadata:
		if not 0 <= index < self._max_index:
			raise IndexError(
				f"RecordMetadata object index is out of bounds; Got index {index},"
				f" allowed indices should be in [0, {self._max_index}]"
			)

		record_key = self._record_keys[index]
		rng = np.random.Generator(np.random.Philox(key=self._seed + index))
		return RecordMetadata(index, record_key, rng)

	def __len__(self) -> int:
		return self._max_index

class ImageTransform(RandomMapTransform):
	def __init__(self, transform_fn):
		self.transform_fn = transform_fn

	def random_map(self, data: tuple[np.ndarray, np.ndarray], rng: np.random.Generator) -> tuple[jax.Array, jax.Array]:
		images, labels = data
		images, labels = jnp.array(images), jnp.array(labels)

		if len(images.shape) == 3:
			images = images[:, :, :, None]

		key = jax.random.PRNGKey(rng.integers(0, 2**32))
		images = self.transform_fn(images, key)
		return images, labels