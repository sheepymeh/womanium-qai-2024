import jax
from jax import numpy as jnp
from flax import linen as nn
from transformers import AutoProcessor

def load_hf(model_class, model_name):
	processor = AutoProcessor.from_pretrained(model_name)
	model = model_class.from_pretrained(model_name)
	module = model.module
	variables = model.params
	return module, variables, processor

class Classifier(nn.Module):
	num_classes: int
	backbone: nn.Module

	@nn.compact
	def __call__(self, x: jnp.ndarray):
		x = self.backbone(x).pooler_output
		x = x.reshape((x.shape[0], -1))
		x = nn.Dense(
			self.num_classes, name='head', kernel_init=nn.zeros
		)(x)
		return x

def init_model(
	model_class,
	model_name,
	num_classes,
	rng_key,
):
	backbone, backbone_vars, processor = load_hf(model_class, model_name)
	model = Classifier(num_classes=num_classes, backbone=backbone)
	variables = model.init(rng_key, jnp.empty((1, 224, 224, 3)))
	params = variables['params']
	params['backbone'] = backbone_vars['params']
	batch_stats = variables['batch_stats']
	batch_stats['backbone'] = backbone_vars['batch_stats']
	return model, processor, params, batch_stats
