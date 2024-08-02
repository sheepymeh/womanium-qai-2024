import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
import optax
from transformers import FlaxResNetModel

from tqdm.auto import tqdm, trange
import argparse
from dataset import ImageDataSource, WeightedIndexSampler, ImageTransform, train_transform, val_transform
from grain.python import DataLoader, NoSharding, Batch, SequentialSampler
from utils import Metrics, calc_acc
from model import init_model


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--dataloader_workers', type=int, default=6)
	return parser.parse_args()


def main():
	args = parse_args()
	key = jax.random.PRNGKey(args.seed)


	train_data = ImageDataSource(args.data_dir, 'train')
	train_class_p = 1 / (np.stack([(train_data.labels == i).sum() for i in range(6)]))
	train_data_p = train_class_p[train_data.labels]
	train_data_p /= train_data_p.sum()
	train_steps_per_epoch = len(train_data) // args.batch_size + 1

	val_data = ImageDataSource(args.data_dir, 'test')
	val_sampler = SequentialSampler(
		num_records=len(val_data),
		shard_options=NoSharding(),
		seed=args.seed,
	)
	val_loader = DataLoader(
		data_source=val_data,
		operations=[
			Batch(args.batch_size, False),
			ImageTransform(val_transform),
		],
		sampler=val_sampler,
		worker_count=args.dataloader_workers,
		shard_options=NoSharding(),
	)
	val_steps_per_epoch = len(val_data) // args.batch_size + 1


	key, params_key = jax.random.split(key)
	model, processor, params, batch_stats = init_model(FlaxResNetModel, 'microsoft/resnet-50', 6, params_key)


	lr_schedule = optax.cosine_onecycle_schedule(
		transition_steps=args.num_epochs * train_steps_per_epoch,
		peak_value=args.lr,
		pct_start=.1,
		final_div_factor=1000,
	)
	solver = optax.yogi(lr_schedule)
	solver_state = solver.init(params)


	def forward_and_loss(variables, images, labels, train, rngs=None):
		preds, updates = model.apply(variables, images, mutable=['batch_stats'])
		batch_stats = updates['batch_stats']
		loss = optax.losses.softmax_cross_entropy_with_integer_labels(preds, labels).mean()
		return loss, (preds, batch_stats)


	train_metrics = Metrics(['loss', 'acc'])
	val_metrics = Metrics(['loss', 'acc'])
	for epoch in trange(1, args.num_epochs + 1):
		train_metrics.reset()
		val_metrics.reset()
		key, subkey = jax.random.split(key)


		train_sampler = WeightedIndexSampler(
			weights=train_data_p,
			seed=args.seed + epoch,
			num_epochs=1,
		)
		train_loader = DataLoader(
			data_source=train_data,
			operations=[
				Batch(args.batch_size, False),
				ImageTransform(train_transform),
			],
			sampler=train_sampler,
			worker_count=args.dataloader_workers,
			shard_options=NoSharding(),
		)

		for images, labels in (pbar := tqdm(train_loader, total=train_steps_per_epoch, desc='Training', leave=False)):
			(loss, (preds, batch_stats)), grad = jax.value_and_grad(forward_and_loss, has_aux=True)({'params': params, 'batch_stats': batch_stats}, images, labels, True, subkey)
			updates, solver_state = solver.update(grad['params'], solver_state, params)
			params = optax.apply_updates(params, updates)

			train_metrics.update({
				'loss': loss.item(),
				'acc': calc_acc(preds, labels),
			})
			pbar.set_postfix_str(train_metrics.previous)

		tqdm.write(f'epoch {epoch}: {train_metrics.epoch}')


		for images, labels in (pbar := tqdm(val_loader, total=val_steps_per_epoch, desc='Validation', leave=False)):
			loss, (preds, batch_stats) = forward_and_loss(params, images, labels, False)

			val_metrics.update({
				'loss': loss.item(),
				'acc': calc_acc(preds, labels),
			})
			pbar.set_postfix_str(val_metrics.previous)

		tqdm.write(f'  -> val: {val_metrics.epoch}')

if __name__ == '__main__':
	main()