"""Small W&B integration shared by the standalone experiment scripts."""
import os
import re
from pathlib import Path


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _dotenv_candidates(path=None):
    if path:
        yield Path(path).expanduser()
        return

    seen = set()
    starts = (Path.cwd(), Path(__file__).resolve().parents[1])
    for start in starts:
        for directory in (start, *start.parents):
            candidate = directory / ".env"
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def load_dotenv(path=None):
    """Load the nearest .env without overriding already exported variables."""
    env_path = next((item for item in _dotenv_candidates(path) if item.is_file()), None)
    if env_path is None:
        return None

    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not _ENV_KEY.fullmatch(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    return str(env_path)


def init_wandb(enabled, config, default_project, run_name, env_file="",
               group="", job_type=""):
    """Initialize W&B lazily so non-W&B runs do not require the package."""
    if not enabled:
        return None

    env_path = load_dotenv(env_file or None)
    if env_path:
        print(f"Loaded W&B environment from: {env_path}")

    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        # Match notebook execution: establish explicit session credentials
        # before init instead of relying on the SDK's implicit login path.
        wandb.login(key=api_key)

    project = os.environ.get("WANDB_PROJECT", default_project)
    kwargs = {
        "project": project,
        "name": run_name,
        "config": config,
    }
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        kwargs["entity"] = entity
    if group:
        kwargs["group"] = group
    if job_type:
        kwargs["job_type"] = job_type
    return wandb.init(**kwargs)
