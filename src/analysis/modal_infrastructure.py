import contextlib
import os
import subprocess
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["huggingface-hub==0.35.1", "msa-pairformer==1.0.1"])
    .env({"HF_HOME": "/data"})
    .add_local_python_source("analysis")
)

volume = modal.Volume.from_name("pairformer-model-cache", create_if_missing=True)


BASE_IMAGE = modal.Image.debian_slim(python_version="3.12")


def _collect_paths(obj) -> set[Path]:
    if isinstance(obj, Path):
        return {obj} if obj.exists() else set()
    elif isinstance(obj, list | tuple):
        paths = set()
        for item in obj:
            paths.update(_collect_paths(item))
        return paths
    elif isinstance(obj, dict):
        paths = set()
        for item in obj.values():
            paths.update(_collect_paths(item))
        return paths
    else:
        return set()


def _resolve_path_args(obj):
    if isinstance(obj, Path):
        return obj.resolve()
    elif isinstance(obj, list | tuple):
        return type(obj)(_resolve_path_args(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: _resolve_path_args(v) for k, v in obj.items()}
    return obj


def use_modal() -> bool:
    return os.environ.get("USE_MODAL", "").strip().lower() in {"1", "true"}


def runnable_on_modal(
    *,
    app_name: str,
    base_image: modal.Image = BASE_IMAGE,
    local_dir: Path | str | None = None,
    auto_detect_files: bool = False,
    ignore_pattern: Sequence[str] | Callable[[Path], bool] = (),
    enable_output: bool = True,
    **remote_fn_kwargs,
):
    """Conditionally run functions on Modal with some basic automatic file handling.

    Enables functions to run either locally or on Modal based on the USE_MODAL
    environment variable (set to either 0 or 1).

    Ideally, the decorated function does not depend on the ability to read data from
    files, since the local and remote have different filesystems, but in practice most
    compute-intensive functions do read data from file. To partially address this, This
    decorator will detect Path objects from within the decorated function's call
    signature and upload them to the remote container if `auto_detect_files` is set to
    True (see also `local_dir`). Uploaded files will exist on the container only for the
    duration of the function call, i.e., there is no persistent storage. For this
    reason, file uploads should only be used for small files or if only a small number
    of function calls are required. Indiscriminate use of file upload features (either
    with `local_dir` or `auto_detect_files`) will cause severe performance issues and
    wasted resources.

    Args:
        app_name:
            Name for the Modal app.
        base_image:
            Base Modal image to use. Defaults to BASE_IMAGE.
        auto_detect_files:
            Whether to automatically detect and upload Path objects from function
            arguments. Path objects must be used (strings are not recognized as paths).
        local_dir:
            Local directory to mount in the container. WARNING: The directory is
            uploaded ephemerally for each function call with no persistent storage. Use
            ignore_pattern to exclude large files/dirs or avoid this parameter entirely
            and instead rely on `auto_detect_files`.
        ignore_pattern:
            Patterns to ignore when mounting local_dir. Can be a sequence of strings or
            a callable that takes a Path and returns bool.
        **remote_fn_kwargs:
            Additional keyword arguments passed to modal.App.function().

    Returns:
        The decorated function that runs locally or on Modal based on USE_MODAL.

    Limitations:
        - Functions should not write to file, since any written files on the remote
          container are irretrievable and will be lost after execution
        - `auto_detect_files` Path argument parsing will fail for complex nested
          structures, generators, or custom objects.

    Example:
        @modal_run_settings(app_name="my-app", gpu="T4", auto_detect_files=True)
        def process_file(input_path: Path) -> str:
            with open(input_path) as f:
                return f.read().upper()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not use_modal():
                return func(*args, **kwargs)

            image = base_image

            resolved_local_dir = None
            if local_dir:
                resolved_local_dir = Path(local_dir).resolve()
                image = image.add_local_dir(
                    local_path=resolved_local_dir,
                    remote_path=str(resolved_local_dir),
                    ignore=ignore_pattern,
                )

            if auto_detect_files:
                paths_to_add = set()
                paths_to_add.update(_collect_paths(args))
                paths_to_add.update(_collect_paths(kwargs))

                for path in paths_to_add:
                    if resolved_local_dir and path.is_relative_to(resolved_local_dir):
                        continue
                    image = image.add_local_file(path, str(path.resolve()))

            app = modal.App(image=image, name=app_name)
            env = app.function(**remote_fn_kwargs)
            modal_func = env(func)

            new_args = tuple([_resolve_path_args(arg) for arg in args])
            new_kwargs = {k: _resolve_path_args(v) for k, v in kwargs.items()}

            output_context = modal.enable_output() if enable_output else contextlib.nullcontext()
            with output_context, app.run():
                return modal_func.remote(*new_args, **new_kwargs)

        return wrapper

    return decorator


@runnable_on_modal(app_name="test-app", gpu="T4")
def _some_function():
    print("Running on:")
    subprocess.run(["uname", "-n"])


if __name__ == "__main__":
    _some_function()
