"""Start, stop, and introspect the two inference backends.

Both engines serve the *same* GGUF file, so any measured difference is the
server, not the weights. The GPU knob is the number of transformer layers
offloaded to Metal: ``-ngl N`` for llama.cpp, ``PARAMETER num_gpu N`` for Ollama.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434"
LLAMACPP_PORT = 8080
LLAMACPP_URL = f"http://127.0.0.1:{LLAMACPP_PORT}"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def wait_http(url: str, timeout: float = 180.0) -> bool:
    """Poll a URL until it answers 200, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(0.5)
    return False


def gguf_path_for(model: str) -> Path:
    """Resolve an Ollama model tag to the GGUF blob on disk.

    ``ollama show --modelfile`` prints a ``FROM <path>`` line pointing at the
    content-addressed blob. Handing that same file to llama-server is what makes
    the comparison apples-to-apples.
    """
    out = _run(["ollama", "show", model, "--modelfile"]).stdout
    for match in re.finditer(r"^FROM\s+(\S.*)$", out, re.MULTILINE):
        candidate = Path(match.group(1).strip())
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"could not find a GGUF blob for {model!r}; is it pulled? (ollama pull {model})"
    )


class LlamaServer:
    """Context manager around a `llama-server` process pinned to one -ngl value."""

    def __init__(self, gguf: Path, ngl: int, ctx: int = 4096, port: int = LLAMACPP_PORT,
                 log_dir: Path | None = None):
        self.gguf, self.ngl, self.ctx, self.port = gguf, ngl, ctx, port
        self.log_dir = log_dir or Path("results")
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> str:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_dir / f"llamacpp-ngl{self.ngl}.log", "w")
        self.proc = subprocess.Popen(
            [
                "llama-server", "-m", str(self.gguf),
                "--host", "127.0.0.1", "--port", str(self.port),
                "-ngl", str(self.ngl), "-c", str(self.ctx),
            ],
            stdout=self._log, stderr=subprocess.STDOUT,
        )
        if not wait_http(f"http://127.0.0.1:{self.port}/health", timeout=300):
            self.__exit__(None, None, None)
            raise RuntimeError(f"llama-server failed to come up (see {self._log.name})")
        return f"http://127.0.0.1:{self.port}/v1"

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            self.proc = None
        try:
            self._log.close()
        except Exception:
            pass
        # Port needs a moment before the next server can bind it.
        time.sleep(2)


def ollama_variant(base_model: str, ngl: int) -> str:
    """Create a derived Ollama model whose only change is the GPU layer count.

    Ollama's OpenAI-compatible endpoint has no field for `num_gpu`, so the knob
    has to be baked into the model. Derived models reuse the base blobs, so this
    costs no extra disk.
    """
    tag = f"bench-ngl{ngl}"
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as fh:
        fh.write(f"FROM {base_model}\nPARAMETER num_gpu {ngl}\n")
        modelfile = fh.name
    res = _run(["ollama", "create", tag, "-f", modelfile])
    Path(modelfile).unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"ollama create {tag} failed: {res.stderr.strip()}")
    return tag


def ollama_unload(model: str) -> None:
    """Evict a model so the next variant loads fresh instead of reusing memory."""
    _run(["ollama", "stop", model])
    time.sleep(2)
