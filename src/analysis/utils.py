from functools import partial

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
progress = partial(tqdm, bar_format=_fmt)
async_gather_with_progress = partial(tqdm_asyncio.gather, bar_format=_fmt)
