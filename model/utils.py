from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class Metric:
	total: float = 0.
	previous: float = 0.
	counter: int = 0

class Metrics:
	def __init__(self, metrics: list[str]) -> None:
		self.keys = metrics
		self.reset()

	def reset(self) -> None:
		self.metrics = {k: Metric() for k in self.keys}

	def update(self, metrics: dict[str, float|int]) -> None:
		for k, v in metrics.items():
			self.metrics[k].total += v
			self.metrics[k].previous = v
			self.metrics[k].counter += 1

	@property
	def epoch_dict(self) -> dict[str, float]:
		return {k: v.total / v.counter for k, v in self.metrics.items()}

	@property
	def epoch(self) -> str:
		return '\t'.join([f'{k}: {v.total / v.counter:.4f}' for k, v in self.metrics.items()])

	@property
	def previous(self) -> str:
		return ', '.join([f'{k}: {v.previous:.4f}' for k, v in self.metrics.items()])

def calc_acc(preds: jnp.ndarray, labels: jnp.ndarray) -> float:
	return (preds.argmax(axis=-1) == labels).mean().item()