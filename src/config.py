import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._create_attributes(config_dict)

    def _create_attributes(self, config_dict: Dict[str, Any], prefix: str = ""):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str = None) -> Config:
    if config_path is None:
        # Default to project root/config.yaml
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.yaml exists in the project root."
        )

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_config() -> Config:
    return load_config()


# Example usage 
if __name__ == "__main__":
    config = get_config()

    print(f"Imbalance Ratios: {config.experiment.imbalance_ratios}")
    print(f"Number of Repetitions: {config.experiment.n_repetitions}")
    print(f"Generators: {config.models.generators}")

    print(f"Random State: {config.get('experiment.random_state', 42)}")

    print(f"Test Size: {config.data_preparation.test_size}")
    print(f"Parallel Jobs: {config.execution.parallel.n_jobs}")
