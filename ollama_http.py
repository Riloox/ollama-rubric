"""Minimal Ollama HTTP client. Stdlib only, so the demo runs anywhere Ollama does.

Real client talks to /api/generate (stream=false). Fake client produces
deterministic synthetic answers derived from the task file, so the scorer and
the table can be exercised without a GPU, a model, or a running server.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 120.0,
                 retries: int = 2):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}{path}", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{path} failed after {self.retries + 1} tries: {last}")

    def generate(self, model: str, prompt: str, temperature: float = 0.0,
                 system: str | None = None, think: bool | None = None,
                 fmt: dict | str | None = None) -> tuple[str, float]:
        """Return (answer, seconds). Temperature 0 by default: an eval that moves is not an eval.

        system: sent as the system prompt when given. think: sent as the `think` field when not
        None; None leaves the field out, so the model's own default applies. fmt: sent as Ollama's
        `format` field (e.g. a JSON schema dict or "json") when not None."""
        t0 = time.monotonic()
        payload = {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if think is not None:
            payload["think"] = think
        if fmt is not None:
            payload["format"] = fmt
        data = self._post("/api/generate", payload)
        return (data.get("response") or "").strip(), time.monotonic() - t0

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.host}{path}", timeout=10) as r:
            return json.loads(r.read().decode())

    def models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def environment(self, models: list[str]) -> dict:
        """What a stranger needs to see why two runs of the same model differ: the Ollama
        version and each model's digest. A failed lookup is recorded as None, never raised."""
        try:
            version = self._get("/api/version").get("version")
        except Exception:
            version = None
        try:
            digests = {m["name"]: m.get("digest") for m in self._get("/api/tags").get("models", [])}
        except Exception:
            digests = {}
        return {"ollama_version": version,
                "model_digests": {m: digests.get(m) or digests.get(m + ":latest") for m in models}}


class Fake:
    """Deterministic stand-in. Correct about 60% of the time, in a fixed pattern per id,
    so both the pass path and the failure path get exercised. Never report these numbers."""

    def __init__(self, seed: str = ""):
        self.seed = seed

    def generate(self, model: str, prompt: str, temperature: float = 0.0) -> tuple[str, float]:
        return "", 0.0  # replaced per-item by FakeForTasks

    def models(self) -> list[str]:
        return ["fake-a", "fake-b"]


class FakeForTasks:
    """Needs the task row to fake a plausible answer, so rubric.py builds one per model."""

    def __init__(self, model: str, salt: str = ""):
        self.model = model
        self.salt = salt

    def answer(self, item: dict) -> tuple[str, float]:
        h = hashlib.sha256((self.salt + self.model + item["id"]).encode()).digest()
        good = h[0] % 10 < 6
        fix = (item.get("fix") or "").strip()
        original = ((item.get("check") or {}).get("values") or [""])[0]
        if not fix:  # correction_ok items: the right move is to confirm, not to fix
            if good:
                text = f"**맞아요.** \"{original}\" 그대로 두시면 됩니다."
            else:
                text = f"**고친 문장:** {original.replace('a', 'e', 1)}\n\n조금 다듬었어요."
        elif good:
            text = f"**고친 문장:** {fix}\n\n아주 잘 쓰셨어요. 조금만 다듬으면 됩니다."
        else:
            text = f"**고친 문장:** {fix.replace('a', 'e', 1)}\n\n잘 했어요."
        return text, 0.001 * (h[1] + 1)

    def models(self) -> list[str]:
        return [self.model]


def make_client(host: str | None, fake_model: str | None):
    """Returns (client, is_fake). Fake implies a model name, since nothing is really served."""
    if fake_model:
        return FakeForTasks(fake_model), True
    return Client(host or "http://127.0.0.1:11434"), False
