import os
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch


class AverageValue(object):
    """
    This class can manage the average value of float numbers.
    With update function, you can add new float numbers and you can get the average value with avg property.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._val: float = 0
        self._avg: float = 0
        self._sum: float = 0
        self._count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def count(self) -> int:
        return self._count

    @property
    def avg(self) -> float:
        return self._avg


class Logger(object):
    """
    This class manages logging.
    Args:
        name (str|None):    The name of the project calling this logger. If you set None, this class just print logs.
        save_dir:           The path to the directory to save the logs in. If you set None, this class just print logs.
    """

    def __init__(self, name: Optional[str] = None, save_dir: Optional[str] = None) -> None:
        if name and save_dir:
            self.save_path: Optional[str] = os.path.join(save_dir, name)
            self.write(f"Logging started at {str(datetime.now())}.\n" +
                       f"They will be saved in {self.save_path}")
        else:
            self.save_path = None

    def write(self, log: str) -> None:
        if self.save_path:
            open(self.save_path, 'a').write(log)
        print(log)


def fix_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
