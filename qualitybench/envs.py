"""The pinned harness environments, installed once by `qualitybench setup` (or on first use).

lm-eval + evalplus, BFCL and LiveCodeBench pin incompatible dependencies, so each gets its own
locked environment under $QUALITYBENCH_HOME (default ~/.cache/qualitybench):

  venvs/lmeval   lm-eval 0.4.13, evalplus 0.3.1, transformers (locks/lmeval.txt)
  venvs/bfcl     bfcl-eval 2026.3.23 (locks/bfcl.txt)
  LiveCodeBench  docker image qualitybench-lcb:28fef95e (docker/lcb), or venvs/lcb with --sandbox native
  hf/            Hugging Face datasets, fetched at setup so a run never downloads mid-way
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

from .progress import log

LCB_REPO = "https://github.com/LiveCodeBench/LiveCodeBench"
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_IMAGE = "qualitybench-lcb:28fef95e"
EVALPLUS_IMAGE = "ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740"
LCB_NATIVE_PINS = ["datasets==3.6.0", "numpy<2.3", "pebble", "tqdm", "openai==1.59.6", "anthropic==0.42.0",
                   "cohere==5.13.6", "google-genai==0.6.0", "mistralai==0.4.2", "together==1.3.14"]


def home() -> Path:
    return Path(os.environ.get("QUALITYBENCH_HOME") or Path.home() / ".cache" / "qualitybench")


def pkg_file(*parts: str) -> Path:
    return Path(str(resources.files("qualitybench").joinpath(*parts)))


def venv_python(name: str) -> Path:
    return home() / "venvs" / name / "bin" / "python"


def venv_bin(name: str, exe: str) -> Path:
    return home() / "venvs" / name / "bin" / exe


def harness_env(extra: dict | None = None) -> dict:
    """Environment for harness subprocesses: shared HF and evalplus caches under the QualityBench home."""
    h = home()
    env = dict(os.environ)
    env.update({"HF_HOME": str(h / "hf"), "XDG_CACHE_HOME": str(h / "xdg"), "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    env.pop("QUALITYBENCH_API_KEY", None)  # harnesses talk to the local proxy, which adds the key
    env.update(extra or {})
    return env


def uv() -> str:
    exe = shutil.which("uv")
    if not exe:
        raise SystemExit("qualitybench setup needs uv (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh")
    return exe


def _run(cmd: list, **kw) -> None:
    # An inline script logs as its first line only, so it doesn't fill the console.
    shown = [a if "\n" not in a else a.strip().splitlines()[0] + " …" for a in map(str, cmd)]
    log("$ " + " ".join(shown))
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def _stamp(name: str) -> Path:
    return home() / "venvs" / name / ".qualitybench-ok"


def _lock_id(lock: Path) -> str:
    import hashlib
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def ensure_venv(name: str) -> None:
    lock = pkg_file("locks", f"{name}.txt")
    st = _stamp(name)
    if st.is_file() and st.read_text().strip() == _lock_id(lock):
        return
    d = home() / "venvs" / name
    log(f"setup: installing the {name} harness environment into {d} (one time)")
    _run([uv(), "venv", "-q", "--clear", "--python", "3.12", d])
    _run([uv(), "pip", "install", "-q", "--no-deps", "--python", d / "bin" / "python", "-r", lock])
    st.write_text(_lock_id(lock))


def ensure_lcb_source() -> Path:
    src = home() / "lcb-src"
    if (src / ".qualitybench-ok").is_file():
        return src
    log(f"setup: fetching LiveCodeBench {LCB_COMMIT[:8]}")
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    _run(["git", "init", "-q", src])
    _run(["git", "-C", src, "fetch", "-q", "--depth", "1", LCB_REPO, LCB_COMMIT])
    _run(["git", "-C", src, "checkout", "-q", "FETCH_HEAD"])
    (src / ".qualitybench-ok").write_text(LCB_COMMIT)
    return src


def docker_image_exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def ensure_lcb(sandbox: str) -> None:
    src = ensure_lcb_source()
    if sandbox == "native":
        st = _stamp("lcb")
        if st.is_file():
            return
        d = home() / "venvs" / "lcb"
        _run([uv(), "venv", "-q", "--clear", "--python", "3.11", d])
        py = d / "bin" / "python"
        _run([uv(), "pip", "install", "-q", "--python", py, "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
        _run([uv(), "pip", "install", "-q", "--python", py, *LCB_NATIVE_PINS])
        _run([uv(), "pip", "install", "-q", "--python", py, "--no-deps", "-e", src])
        st.write_text(LCB_COMMIT)
        return
    if docker_image_exists(LCB_IMAGE):
        return
    ctx = home() / "lcb-docker"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(parents=True)
    shutil.copy(pkg_file("docker", "lcb", "Dockerfile"), ctx / "Dockerfile")
    shutil.copytree(src, ctx / "src", ignore=shutil.ignore_patterns(".git"))
    log(f"setup: building {LCB_IMAGE}")
    _run(["docker", "build", "-q", "-t", LCB_IMAGE, ctx])


def ensure_evalplus_image() -> None:
    if not docker_image_exists(EVALPLUS_IMAGE):
        _run(["docker", "pull", "-q", EVALPLUS_IMAGE])


PREFETCH = r"""
import sys
want = set(sys.argv[1].split(","))
from datasets import load_dataset
if "gsm8k" in want: print("gsm8k", len(load_dataset("openai/gsm8k", "main", split="test")))
if "ifeval" in want: print("ifeval", len(load_dataset("google/IFEval", split="train")))
if "mmlu_pro" in want: print("mmlu_pro", len(load_dataset("TIGER-Lab/MMLU-Pro", split="test")))
if want & {"humaneval", "mbpp"}:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    print("humaneval+", len(get_human_eval_plus()), "mbpp+", len(get_mbpp_plus()))
"""

LCB_PREFETCH = ("from lcb_runner.benchmarks.code_generation import load_code_generation_dataset;"
                "print('lcb v6', len(load_code_generation_dataset('v6')))")


def lcb_docker_base(net: str, mounts: list[tuple[Path, str, str]], extra: list[str] | None = None) -> list[str]:
    cmd = ["docker", "run", "--rm", "--network", net, "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "HOME=/tmp", "-e", "HF_HOME=/hf"]
    for src, dst, mode in [(home() / "hf", "/hf", "rw"), *mounts]:
        cmd += ["-v", f"{src}:{dst}:{mode}"]
    return cmd + (extra or [])


def setup(tasks: list[str], sandbox: str = "docker") -> None:
    """Install what the given tasks need and fetch their datasets."""
    (home() / "hf").mkdir(parents=True, exist_ok=True)
    (home() / "xdg").mkdir(parents=True, exist_ok=True)
    need_lm = set(tasks) & {"gsm8k", "ifeval", "mmlu_pro", "humaneval", "mbpp", "logprobs", "needle"}
    if need_lm:
        ensure_venv("lmeval")
        _run([venv_python("lmeval"), "-c", PREFETCH, ",".join(sorted(need_lm))], env=harness_env())
    if "bfcl" in tasks:
        ensure_venv("bfcl")
    if set(tasks) & {"humaneval", "mbpp"} and sandbox == "docker":
        ensure_evalplus_image()
    if "lcb" in tasks:
        ensure_lcb(sandbox)
        if sandbox == "native":
            _run([home() / "venvs" / "lcb" / "bin" / "python", "-c", LCB_PREFETCH], cwd=home() / "lcb-src", env=harness_env())
        else:
            _run(lcb_docker_base("bridge", []) + [LCB_IMAGE, "python", "-c", LCB_PREFETCH], env=harness_env())


def python_for_helpers() -> str:
    return sys.executable
