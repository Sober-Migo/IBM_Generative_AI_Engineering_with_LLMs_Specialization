#!/usr/bin/env python3
"""
IBM Generative AI Engineering with LLMs Specialization — Interactive Setup
---------------------------------------------------------------------------
Courses 1–7 · Python 3.10 · CUDA 12.1 (optional GPU)

This script helps you:
  1. Create a Conda or venv environment (recommended: Conda, name llms_env)
  2. Install the locked package stack (PyTorch, Transformers, PEFT, LangChain, …)
  3. Set useful environment variables:
       - KERAS_BACKEND=torch
       - HF_HOME / TORCH_HOME (optional, for cache flexibility on small disks)
  4. Register a Jupyter kernel
  5. Optionally download large model/dataset assets (not stored in Git)

Stable environment (do not change pins casually):
  - Python 3.10
  - torch 2.2.2+cu121 (or CPU if no NVIDIA GPU)
  - transformers 4.43.4 + matching peft / trl / accelerate

Works on Windows, macOS, and Linux.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
REQUIRED_PYTHON_STR = "3.10"
DEFAULT_CONDA_ENV = "llms_env"
PROJECT_ROOT = Path(__file__).resolve().parent


def print_header() -> None:
    print("\n" + "=" * 72)
    print("  IBM Generative AI Engineering with LLMs — Environment Setup")
    print("  Courses 1–7 · Python 3.10 · CUDA 12.1 (optional)")
    print("=" * 72 + "\n")


def print_step(msg: str) -> None:
    print(f"\n>>> {msg}")


def ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def run_command(cmd: list, check: bool = True, env: dict | None = None) -> bool:
    try:
        subprocess.run(cmd, check=check, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed ({e.returncode}): {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {cmd[0]}")
        return False


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_python_version() -> None:
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"[WARNING] Running Python {sys.version.split()[0]}")
        print(f"          This specialization is tested with Python {REQUIRED_PYTHON_STR}.x")
        if not ask_yes_no("Continue anyway?", default=False):
            print("Exiting. Please use Python 3.10.")
            sys.exit(1)


def prompt_cache_dirs() -> dict:
    print_step("Optional cache directories (HF_HOME, TORCH_HOME)")
    print("  Hugging Face models and PyTorch hub caches can be large.")
    print("  Leave blank to use defaults (~/.cache/huggingface, ~/.cache/torch).\n")

    env_extra: dict[str, str] = {}

    if ask_yes_no("Set a custom HF_HOME (Hugging Face cache)?", default=False):
        path = input("  HF_HOME path: ").strip().strip('"')
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            env_extra["HF_HOME"] = path
            print(f"  → HF_HOME = {path}")

    if ask_yes_no("Set a custom TORCH_HOME (PyTorch hub cache)?", default=False):
        path = input("  TORCH_HOME path: ").strip().strip('"')
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
            env_extra["TORCH_HOME"] = path
            print(f"  → TORCH_HOME = {path}")

    env_extra["KERAS_BACKEND"] = "torch"
    print("  → KERAS_BACKEND = torch")
    return env_extra


def apply_conda_env_vars(env_name: str, vars_map: dict[str, str]) -> None:
    if not command_exists("conda"):
        return
    print_step(f"Setting Conda env vars on '{env_name}' …")
    for key, value in vars_map.items():
        run_command(
            ["conda", "env", "config", "vars", "set", f"{key}={value}", "-n", env_name],
            check=False,
        )
    print("  (Re-activate the env later so vars take effect: conda deactivate && conda activate", env_name + ")")


def write_dotenv(vars_map: dict[str, str]) -> None:
    dotenv = PROJECT_ROOT / ".env"
    existing: dict[str, str] = {}
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()
    existing.update(vars_map)
    dotenv.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n", encoding="utf-8")
    print(f"  Wrote {dotenv.name} (gitignored).")


def setup_conda() -> tuple[str, str, str]:
    if not command_exists("conda"):
        print("[ERROR] Conda not found. Install Miniconda/Anaconda or choose venv.")
        sys.exit(1)

    env_name = input(f"Conda env name [{DEFAULT_CONDA_ENV}]: ").strip() or DEFAULT_CONDA_ENV
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    exists = env_name in result.stdout

    if exists:
        print(f"Environment '{env_name}' already exists.")
        if ask_yes_no("Remove and recreate it?", default=False):
            run_command(["conda", "env", "remove", "-n", env_name, "-y"])
            exists = False
        else:
            print(f"Using existing env '{env_name}'.")

    if not exists:
        print_step(f"Creating Conda env '{env_name}' with Python {REQUIRED_PYTHON_STR} …")
        ok = run_command(
            ["conda", "create", "-n", env_name, f"python={REQUIRED_PYTHON_STR}", "-c", "conda-forge", "-y"]
        )
        if not ok:
            sys.exit(1)

    if platform.system() == "Windows":
        print_step("Installing Windows C++ runtime helpers (vc, vs2015_runtime) …")
        run_command(
            ["conda", "install", "-n", env_name, "-c", "conda-forge", "pip", "ipykernel", "vc", "vs2015_runtime", "-y"],
            check=False,
        )
    else:
        run_command(
            ["conda", "install", "-n", env_name, "-c", "conda-forge", "pip", "ipykernel", "-y"],
            check=False,
        )

    return "conda", env_name, env_name


def setup_venv() -> tuple[str, str, str]:
    venv_path = PROJECT_ROOT / "venv"
    if venv_path.exists():
        if ask_yes_no("venv already exists. Recreate?", default=False):
            shutil.rmtree(venv_path)
        else:
            print("Using existing venv.")

    if not venv_path.exists():
        print_step("Creating venv …")
        import venv
        venv.create(venv_path, with_pip=True)

    if platform.system() == "Windows":
        python_exe = str(venv_path / "Scripts" / "python.exe")
    else:
        python_exe = str(venv_path / "bin" / "python")
    return "venv", python_exe, str(venv_path)


def pip_cmd(env_type: str, env_id: str) -> list[str]:
    if env_type == "conda":
        return ["conda", "run", "-n", env_id, "python", "-m", "pip"]
    return [env_id, "-m", "pip"]


def python_cmd(env_type: str, env_id: str) -> list[str]:
    if env_type == "conda":
        return ["conda", "run", "-n", env_id, "python"]
    return [env_id]


def install_stack(env_type: str, env_id: str, use_cuda: bool) -> None:
    pip = pip_cmd(env_type, env_id)
    print_step("Upgrading pip …")
    run_command(pip + ["install", "--upgrade", "pip"], check=False)

    print_step("Installing Jupyter + helpers …")
    run_command(pip + ["install", "jupyterlab", "ipywidgets", "hf_xet", "ipykernel"], check=False)

    if use_cuda:
        print_step("Installing PyTorch 2.2.2 + CUDA 12.1 …")
        run_command(
            pip
            + [
                "install",
                "torch==2.2.2+cu121",
                "torchvision==0.17.2+cu121",
                "torchaudio==2.2.2+cu121",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu121",
                "--no-deps",
            ]
        )
    else:
        print_step("Installing PyTorch 2.2.2 (CPU) …")
        run_command(
            pip
            + [
                "install",
                "torch==2.2.2",
                "torchvision==0.17.2",
                "torchaudio==2.2.2",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )

    print_step("Installing torchtext / torchdata (CPU wheels) …")
    run_command(
        pip
        + [
            "install",
            "torchtext==0.17.2+cpu",
            "torchdata==0.7.1",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "--no-deps",
        ],
        check=False,
    )

    print_step("Core data science stack (NumPy 1.x locked) …")
    run_command(
        pip
        + [
            "install",
            "numpy==1.26.4",
            "pandas<2.2.0",
            "scipy<1.13.0",
            "scikit-learn",
            "portalocker>=2.8.2",
            "requests>=2.32.2",
        ]
    )

    print_step("Classical NLP + visualization …")
    run_command(
        pip
        + [
            "install",
            "spacy",
            "nltk",
            "gensim",
            "pmdarima",
            "matplotlib",
            "seaborn",
            "plotly==5.22.0",
            "dash==2.9.3",
        ]
    )

    print_step("LLM fine-tuning stack (transformers / peft / trl locked) …")
    run_command(
        pip
        + [
            "install",
            "transformers==4.43.4",
            "tokenizers==0.19.1",
            "peft==0.11.1",
            "trl==0.9.6",
            "accelerate==0.34.0",
            "datasets==2.20.0",
            "evaluate",
            "rouge_score==0.1.2",
            "sacrebleu==2.4.2",
            "torchmetrics",
        ]
    )

    print_step("RAG + LangChain ecosystem (pinned together) …")
    magic_pkg = "python-magic-bin==0.4.14" if platform.system() == "Windows" else "python-magic"
    run_command(
        pip
        + [
            "install",
            "langchain==0.2.7",
            "langchain-core==0.2.43",
            "langsmith==0.1.147",
            "langchain-community==0.2.7",
            "langchain-experimental==0.0.62",
            "langchain-chroma==0.1.1",
            "langchain-huggingface==0.0.3",
            "huggingface_hub==0.24.0",
            "langchain-ibm>=0.1.4,<0.2.0",
            "langchain-openai==0.1.16",
            "sentence-transformers==2.7.0",
            "langchainhub==0.1.17",
            "chromadb==0.4.24",
            "faiss-cpu",
            "pypdf==4.2.0",
            "pdfplumber==0.9.0",
            "PyPDF2",
            "pymupdf==1.23.26",
            "unstructured[md,csv,doc]==0.14.9",
            magic_pkg,
            "arxiv>=2.1.3",
            "docx2txt==0.8",
            "fpdf==1.7.2",
            "lxml==5.2.2",
            "Levenshtein",
            "rich>=13.8.0",
            "tenacity",
            "posthog==3.5.0",
            "wget",
            "python-dotenv",
        ]
    )

    print_step("bitsandbytes + ibm-watsonx-ai …")
    run_command(pip + ["install", "bitsandbytes==0.43.1", "--no-deps"], check=False)
    run_command(pip + ["install", "ibm-watsonx-ai"], check=False)

    print_step("Re-lock NumPy 1.26.4 …")
    run_command(pip + ["install", "numpy==1.26.4"], check=False)

    print_step("spaCy language models …")
    py = python_cmd(env_type, env_id)
    run_command(py + ["-m", "spacy", "download", "en_core_web_sm"], check=False)
    run_command(py + ["-m", "spacy", "download", "de_core_news_sm"], check=False)

    print_step("Register Jupyter kernel …")
    display = "Python (llms_env)"
    run_command(
        py + ["-m", "ipykernel", "install", "--user", f"--name={DEFAULT_CONDA_ENV}", f"--display-name={display}"],
        check=False,
    )


def verify(env_type: str, env_id: str) -> None:
    print_step("Verification import check …")
    code = (
        "import torch, transformers, peft, langchain, langchain_core, langchain_community, "
        "langchain_experimental, dotenv, sentence_transformers; "
        "print('--- Verification ---'); "
        "print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); "
        "print('Transformers:', transformers.__version__); "
        "print('LangChain:', langchain.__version__); "
        "print('KERAS_BACKEND:', __import__('os').environ.get('KERAS_BACKEND')); "
        "print('HF_HOME:', __import__('os').environ.get('HF_HOME')); "
        "print('TORCH_HOME:', __import__('os').environ.get('TORCH_HOME')); "
        "print('OK')"
    )
    run_command(python_cmd(env_type, env_id) + ["-c", code], check=False)


def main() -> None:
    print_header()
    check_python_version()

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python (this process): {sys.version.split()[0]}")
    print(f"Project: {PROJECT_ROOT}")
    conda_ok = command_exists("conda")
    print(f"Conda available: {'Yes' if conda_ok else 'No'}\n")

    env_vars = prompt_cache_dirs()
    write_dotenv(env_vars)

    print("\nEnvironment type:")
    print("  1. Conda  (recommended — matches course notes, env name llms_env)")
    print("  2. venv")
    print("  3. Exit")
    choice = input("Choose [1]: ").strip() or "1"
    if choice == "3":
        sys.exit(0)
    if choice == "2":
        env_type, env_id, label = setup_venv()
    else:
        if not conda_ok:
            print("Conda not found; falling back to venv.")
            env_type, env_id, label = setup_venv()
        else:
            env_type, env_id, label = setup_conda()
            apply_conda_env_vars(env_id, env_vars)

    use_cuda = ask_yes_no(
        "Install CUDA 12.1 GPU build of PyTorch? (needs NVIDIA driver; else choose No for CPU)",
        default=True if platform.system() == "Windows" else False,
    )

    install_stack(env_type, env_id, use_cuda=use_cuda)
    verify(env_type, env_id)

    downloader = PROJECT_ROOT / "download_assets.py"
    if downloader.exists() and ask_yes_no(
        "\nDownload large model/dataset assets now? (not stored in Git; required for several labs)",
        default=True,
    ):
        run_command(python_cmd(env_type, env_id) + [str(downloader)], check=False)

    print("\n" + "=" * 72)
    print("  Setup complete")
    print("=" * 72)
    if env_type == "conda":
        print(f"  Activate:  conda activate {env_id}")
        print("  (If you set env vars, run: conda deactivate && conda activate", env_id + ")")
    else:
        if platform.system() == "Windows":
            print(r"  Activate:  venv\Scripts\activate")
        else:
            print("  Activate:  source venv/bin/activate")
    print("  Jupyter:   jupyter lab")
    print("  Assets:    python download_assets.py")
    print("\nHappy learning!\n")


if __name__ == "__main__":
    main()
