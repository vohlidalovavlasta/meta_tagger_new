import json
from typing import Any, Dict


class HParams:
  """Lightweight replacement for tf.contrib.training.HParams."""

  def __init__(self, **kwargs):
    super().__setattr__('_values', {})
    for key, value in kwargs.items():
      self.add_hparam(key, value)

  def add_hparam(self, name: str, value: Any):
    self._values[name] = value
    super().__setattr__(name, value)

  def parse_json(self, json_str: str):
    values = json.loads(json_str)
    for key, value in values.items():
      self.add_hparam(key, value)
    return self

  def values(self) -> Dict[str, Any]:
    return dict(self._values)

  def __setattr__(self, name: str, value: Any):
    if name.startswith('_'):
      super().__setattr__(name, value)
      return
    self.add_hparam(name, value)

  def __getattr__(self, name: str) -> Any:
    if name in self._values:
      return self._values[name]
    raise AttributeError(f"HParams has no attribute '{name}'")
