"""
Importing rag must not construct the OpenAI client.

The retrieval half of the evaluation is the free, deterministic half: it calls
fetch_context and never generates an answer. Building ChatOpenAI at import time
made that half require credentials it never uses, and made the module
unimportable for anyone who cloned the repo without a key.

Checked in a subprocess so the result cannot depend on what other tests have
already imported or on a .env sitting in the working directory.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(code: str) -> subprocess.CompletedProcess:
    # A placeholder key so constructing the client works on a machine with no
    # .env. Nothing here makes a network call; a real .env still wins, because
    # rag.py calls load_dotenv(override=True).
    env = os.environ | {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key-not-used")}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )


def test_importing_rag_does_not_build_the_client():
    result = run("import rag; assert rag._llm is None; print('ok')")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_the_retrieval_eval_does_not_build_the_client():
    """The path that actually matters: the eval imports rag for fetch_context."""
    result = run(
        "import evaluation.eval_retrieval as e; "
        "import rag; assert rag._llm is None; print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_get_llm_returns_the_same_instance_each_time():
    """Lazy must still mean built once, not rebuilt per question."""
    result = run(
        "import rag; a = rag.get_llm(); b = rag.get_llm(); "
        "assert a is b; assert rag._llm is a; print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
