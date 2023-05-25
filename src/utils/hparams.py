from dataclasses import dataclass
import json

@dataclass
class Hparams:
    batch_size: int = 256
    epochs: int = 256

    lr: float = 1e-3
    beta1: float = 0.9
    grad_clipping: float = 1.0

    knn: int = 32
    features: int = 256

    vector_fields_type: str = "grad"
    vector_fields_normalize: bool = True
    vector_fields_edges: str = ""
    vector_fields_triplets: str = "n_ij|n_ik|angle"

    model:str = "gemsnet"

    layers: int = 3

    train_pos: bool = True

    @property
    def vector_fields(self):
        def split(s, delimiter):
            if len(s) > 0:
                return s.split(delimiter)
            return []

        return {
            "type": self.vector_fields_type,
            "normalize": self.vector_fields_normalize,
            "edges": split(self.vector_fields_edges, "|"),
            "triplets": split(self.vector_fields_triplets, "|"),
        }

    def from_json(self, file_name):
        with open(file_name, "r") as fp:
            hparams = json.load(fp)

        for key, value in hparams.items():
            assert key in self.__dict__

            self.__dict__[key] = value

    def to_json(self, file_name):
        with open(file_name, "w") as fp:
            json.dump(self.__dict__, fp, indent=4)

    def dict(self):
        return self.__dict__
