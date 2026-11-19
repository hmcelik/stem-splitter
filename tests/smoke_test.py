"""End-to-end smoke test: separates a short synthetic clip and checks that the stems
plus the residual null against the input. Downloads the model on first run.

    python tests/smoke_test.py [preset]        (default preset: demucs4 - fastest on CPU)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemsplitter import engine as eng  # noqa: E402


def make_clip(path: Path, sr: int = 48000, seconds: float = 8.0):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(0)
    bass = 0.3 * np.sin(2 * np.pi * 55 * t)
    hats = 0.1 * rng.standard_normal(t.size) * (np.sin(2 * np.pi * 4 * t) > 0.9)
    f0 = 220 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = np.cumsum(2 * np.pi * f0 / sr)
    voice = 0.15 * sum(np.sin(k * phase) / k for k in range(1, 12))
    mix = np.stack([bass + hats + voice, bass + 0.7 * hats + voice], axis=1)
    sf.write(str(path), mix.astype(np.float32), sr, subtype="FLOAT")


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "demucs4"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "clip v1.2 (test).wav"          # dots in the name on purpose
        make_clip(src)
        job = eng.Job(src, td / "out", preset=preset, quality="Standard", residual="file")
        res = eng.Engine().run(job, lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True))

        x, sr = sf.read(str(src), dtype="float32")
        expected = set(eng.PRESETS[preset].stems) | {"residual"}
        assert set(res.stems) == expected, f"stems {set(res.stems)} != {expected}"
        total = 0
        for name, p in res.stems.items():
            assert p.exists() and p.name.endswith(f" - {name}.wav"), p
            y, ysr = sf.read(str(p), dtype="float32")
            assert ysr == sr and y.shape == x.shape, (name, ysr, y.shape)
            total = total + y
        err = float(np.max(np.abs(x - total)))
        print(f"null test: max error {err:.2e}")
        assert err < 1e-4, "stems + residual do not sum to the input"
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
