from functools import partial

from rich.console import Console
from tqdm import tqdm

_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
progress = partial(tqdm, bar_format=_fmt)

console = Console()
