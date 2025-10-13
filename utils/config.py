import yaml
from types import SimpleNamespace

def load_config(path):
    """
    Charge un fichier YAML et le convertit en objet accessible par attributs.
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_ns(x) for x in d]
        return d

    return dict_to_ns(cfg_dict)
