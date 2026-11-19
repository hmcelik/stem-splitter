"""
Stem separation engine.

Wraps `audio-separator` (UVR model zoo) but takes over all audio I/O so that:
  * no peak normalisation is applied to the mix or the stems (stems keep their
    true level relative to the original song and can be summed back),
  * stems are written at the ORIGINAL sample rate of the song (models run at
    44.1 kHz internally; we resample back with SoX VHQ),
  * output can be 32-bit float WAV (no clipping, no dither), 24-bit WAV or 24-bit FLAC.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

log = logging.getLogger("stemsplitter")

MODEL_SR = 44100

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(Path.home(), ".local", "share")
    p = Path(base) / "StemSplitter"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(os.environ.get("STEMSPLITTER_MODELS", app_data_dir() / "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_ffmpeg() -> str:
    """audio-separator requires an `ffmpeg` executable on PATH. If none is installed,
    use the binary shipped with the imageio-ffmpeg wheel."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found. Install it (winget install Gyan.FFmpeg) "
                           "or `pip install imageio-ffmpeg`.") from e
    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bin_dir = app_data_dir() / "bin"
    bin_dir.mkdir(exist_ok=True)
    dst = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not dst.exists():
        shutil.copy2(src, dst)
        if os.name != "nt":
            dst.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(dst)


# --------------------------------------------------------------------------------------
# Models / presets
# --------------------------------------------------------------------------------------

SW_6STEM = "BS-Roformer-SW.ckpt"                # vocals, drums, bass, guitar, piano, other
DEMUCS_FT = "htdemucs_ft.yaml"                  # vocals, drums, bass, other

# Dedicated vocal models usable in "Maximum quality" mode (filename -> label)
VOCAL_MODELS: dict[str, str] = {
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": "BS-RoFormer 1297 (viperx) - best SDR, clean",
    "bs_roformer_vocals_resurrection_unwa.ckpt": "BS-RoFormer Resurrection (unwa)",
    "mel_band_roformer_vocals_becruily.ckpt": "Mel-RoFormer Vocals (becruily) - fuller sound",
    "mel_band_roformer_vocals_fv4_gabox.ckpt": "Mel-RoFormer Vocals FV4 (Gabox)",
}
DEFAULT_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Second-stage models that split an existing stem further
KARAOKE_MODEL = "bs_roformer_karaoke_frazer_becruily.ckpt"           # vocals -> lead / backing
DEREVERB_MODEL = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"  # vocal -> dry / reverb
DRUMSEP_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"                 # drums -> kit pieces
DRUM_PIECES = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hi-hat",
               "ride": "ride", "crash": "crash"}


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    stems: tuple[str, ...]


PRESETS: dict[str, Preset] = {p.key: p for p in [
    Preset("max6", "6 stems - Maximum quality",
           "Dedicated vocal model first, then BS-RoFormer SW on the instrumental. Slowest, best.",
           ("vocals", "drums", "bass", "guitar", "piano", "other")),
    Preset("sw6", "6 stems - BS-RoFormer SW",
           "Single pass, one model for all six stems. Great quality, ~2x faster than Maximum.",
           ("vocals", "drums", "bass", "guitar", "piano", "other")),
    Preset("demucs4", "4 stems - HTDemucs FT",
           "Classic Demucs fine-tuned. Lower VRAM, works OK on CPU.",
           ("vocals", "drums", "bass", "other")),
    Preset("vox2", "2 stems - Vocals / Instrumental",
           "Vocal model only; instrumental is the exact remainder (mix - vocals).",
           ("vocals", "instrumental")),
]}

QUALITY = {
    # label: (roformer overlap, demucs shifts)
    "Standard": (4, 1),
    "High": (8, 2),
    "Extreme": (16, 5),
}

OUTPUT_FORMATS = {
    "WAV 32-bit float": ("wav", "FLOAT"),
    "WAV 24-bit": ("wav", "PCM_24"),
    "FLAC 24-bit": ("flac", "PCM_24"),
    "WAV 16-bit": ("wav", "PCM_16"),
}


@dataclass
class Job:
    input_path: Path
    output_dir: Path
    preset: str = "max6"
    vocal_model: str = DEFAULT_VOCAL_MODEL
    quality: str = "High"
    output_format: str = "WAV 32-bit float"
    residual: str = "none"          # "none" | "file" | "other"  (what to do with mix - sum(stems))
    use_directml: bool = False
    # extra splits
    split_vocals: bool = False      # vocals -> lead vocals + backing vocals
    dereverb: bool = False          # lead vocals (or vocals) -> dry + reverb
    split_drums: bool = False       # drums -> kick, snare, toms, hi-hat, ride, crash
    keep_full: bool = True          # also write the stems that were split further


@dataclass
class Result:
    stems: dict[str, Path] = field(default_factory=dict)
    full_stems: set[str] = field(default_factory=set)   # names in `stems` that were split further
    sample_rate: int = 0
    residual_db: float = float("nan")   # level of (mix - sum of stems) relative to mix, dB


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 array [channels, samples] at native sample rate."""
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(data.T), sr
    except Exception:
        pass
    # Fallback: decode anything (mp3/m4a/aac/ogg/...) with ffmpeg to float WAV.
    ffmpeg = ensure_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "decoded.wav"
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(tmp)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
        if r.returncode != 0:
            raise RuntimeError(f"Could not decode {path.name}: {r.stderr.strip()}")
        data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    y = soxr.resample(x.T, sr_in, sr_out, quality="VHQ")
    return np.ascontiguousarray(y.T.astype(np.float32))


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[1] == n:
        return x
    if x.shape[1] > n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def write_audio(path: Path, x: np.ndarray, sr: int, fmt: str) -> Path:
    ext, subtype = OUTPUT_FORMATS[fmt]
    # Append the extension instead of Path.with_suffix(): song titles often contain dots
    # ("Pt.2", "feat.") and with_suffix would cut the name off at the last dot.
    if path.suffix.lower() != "." + ext:
        path = path.parent / (path.name + "." + ext)
    data = x.T
    if subtype != "FLOAT":
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            log.warning("%s peaks at %.2f dBFS - clipped in fixed-point output. "
                        "Use 32-bit float to keep it intact.", path.name, 20 * np.log10(peak))
        data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data, sr, subtype=subtype, format=ext.upper())
    return path


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


# --------------------------------------------------------------------------------------
# audio-separator integration
# --------------------------------------------------------------------------------------

_patch_lock = threading.Lock()
_patched = False
_capture: dict[str, np.ndarray] = {}


def _patch_audio_separator() -> None:
    """Disable the library's peak normalisation and capture stems in memory instead of
    letting it write (it writes 16-bit via pydub by default)."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        try:
            from audio_separator.separator.uvr_lib_v5 import spec_utils
            from audio_separator.separator.common_separator import CommonSeparator
        except ModuleNotFoundError as e:
            raise RuntimeError(f"Missing Python package '{e.name}'. Run setup.bat again to "
                               "install the updated requirements.") from e

        def _no_normalize(wave, max_peak=1.0, min_peak=None):
            return np.asarray(wave)

        def _capture_final(self, stem_path, source, stem_name):
            _capture[str(stem_name).strip().lower()] = np.array(source, dtype=np.float32, copy=True)
            return {stem_name: source}

        spec_utils.normalize = _no_normalize
        CommonSeparator.final_process = _capture_final
        _patched = True


class Engine:
    """Keeps one Separator around so a loaded model is reused between songs."""

    def __init__(self, log_callback: Callable[[str], None] | None = None):
        self._sep = None
        self._sep_key = None
        self._loaded_key = None
        self._log_cb = log_callback
        self._cancel = threading.Event()

    # -- public ---------------------------------------------------------------------
    def cancel(self):
        self._cancel.set()

    @staticmethod
    def device_info() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f"CUDA - {name} ({vram:.0f} GB)"
            try:
                import torch_directml  # noqa: F401
                return "DirectML GPU"
            except ImportError:
                pass
            return "CPU (slow - an NVIDIA GPU is strongly recommended)"
        except Exception as e:  # pragma: no cover
            return f"unknown ({e})"

    def run(self, job: Job, progress: Callable[[float, str], None] | None = None) -> Result:
        self._cancel.clear()
        prog = progress or (lambda f, m: None)
        preset = PRESETS[job.preset]

        prog(0.02, "Loading audio")
        mix, sr = load_audio(job.input_path)
        n_orig = mix.shape[1]
        orig_channels = mix.shape[0]
        if orig_channels == 1:
            mix = np.vstack([mix, mix])
        elif orig_channels > 2:
            log.warning("%d-channel file - downmixing to stereo", orig_channels)
            mix = np.vstack([mix[0::2].mean(0), mix[1::2].mean(0)])

        # Models are trained on music around typical mastering levels. Scale the input to
        # a fixed peak (pure float gain, lossless) and undo it on the way out.
        peak = float(np.max(np.abs(mix))) or 1.0
        gain = 0.9 / peak
        mix44 = resample(mix, sr, MODEL_SR) * gain

        self._n_passes = self._count_passes(job, preset)
        self._pass = 0
        overlap, shifts = QUALITY[job.quality]
        stems44 = self._separate(job, preset, mix44, prog)

        # layout: ordered (name, kind) where kind is "leaf" (sums to the mix) or "full"
        # (a stem that was split further and is only written as a convenience copy).
        layout: list[tuple[str, str]] = [(k, "leaf") for k in stems44]
        arrays: dict[str, np.ndarray] = dict(stems44)

        if job.split_vocals and "vocals" in arrays:
            self._stage(prog, "Lead / backing vocals")
            lead = self._target(KARAOKE_MODEL, arrays["vocals"], job, overlap, shifts, ("vocals", "lead"))
            self._replace(layout, arrays, "vocals",
                          {"lead vocals": lead, "backing vocals": arrays["vocals"] - lead}, job.keep_full)

        if job.dereverb:
            key = next((k for k in ("lead vocals", "vocals") if (k, "leaf") in layout), None)
            if key:
                self._stage(prog, f"Removing reverb from {key}")
                dry = self._target(DEREVERB_MODEL, arrays[key], job, overlap, shifts, ("noreverb", "dry"))
                self._replace(layout, arrays, key,
                              {f"{key} dry": dry, f"{key} reverb": arrays[key] - dry}, job.keep_full)

        if job.split_drums and "drums" in arrays:
            self._stage(prog, "Drum kit (kick, snare, toms, hi-hat, ride, crash)")
            d = self._run_model(DRUMSEP_MODEL, arrays["drums"], job, overlap, shifts)
            pieces = {DRUM_PIECES[k]: d[k] for k in DRUM_PIECES if k in d}
            # What DrumSep leaves unassigned is still drums: give it to the stem it sounds
            # most like would be guesswork, so it goes to the global leftover.
            self._replace(layout, arrays, "drums", pieces, job.keep_full)

        prog(0.92, "Writing stems")
        job.output_dir.mkdir(parents=True, exist_ok=True)
        base = job.input_path.stem
        result = Result(sample_rate=sr)

        # Back to the original sample rate and level.
        native: dict[str, np.ndarray] = {}
        for name, _ in layout:
            self._check_cancel()
            native[name] = fit_length(resample(arrays[name], MODEL_SR, sr), n_orig) / gain

        # Leftover = original mix minus every leaf stem. Computed at the native rate, so it
        # also holds content above 22 kHz (hi-res sources); leaves + leftover null exactly.
        leaves = [n for n, k in layout if k == "leaf"]
        residual = mix - sum(native[n] for n in leaves)
        result.residual_db = db(np.sqrt(np.mean(residual**2))) - db(np.sqrt(np.mean(mix**2)))
        if job.residual == "other":
            key = next((k for k in ("other", "instrumental") if k in leaves), leaves[-1])
            native[key] = native[key] + residual
        elif job.residual == "file":
            native["residual"] = residual
            layout.append(("residual", "leaf"))

        for name, kind in layout:
            s = native[name]
            if orig_channels == 1:
                s = s.mean(axis=0, keepdims=True)
            out = write_audio(job.output_dir / f"{base} - {name}", s, sr, job.output_format)
            result.stems[name] = out
            if kind == "full":
                result.full_stems.add(name)
        prog(1.0, "Done")
        return result

    # -- internals ------------------------------------------------------------------
    def _check_cancel(self):
        if self._cancel.is_set():
            raise Cancelled()

    def _count_passes(self, job: Job, preset: Preset) -> int:
        n = 2 if preset.key == "max6" else 1
        n += bool(job.split_vocals and "vocals" in preset.stems)
        n += bool(job.dereverb and "vocals" in preset.stems)
        n += bool(job.split_drums and "drums" in preset.stems)
        return n

    def _stage(self, prog, msg: str):
        self._check_cancel()
        self._pass += 1
        frac = 0.05 + 0.85 * (self._pass - 1) / max(self._n_passes, 1)
        prog(frac, f"Pass {self._pass}/{self._n_passes} - {msg}")

    @staticmethod
    def _replace(layout, arrays, key, children: dict[str, np.ndarray], keep_full: bool):
        """Replace leaf `key` by `children` (in place, same position)."""
        i = layout.index((key, "leaf"))
        layout[i:i + 1] = ([(key, "full")] if keep_full else []) + [(c, "leaf") for c in children]
        arrays.update(children)

    def _target(self, model_file, x, job, overlap, shifts, names) -> np.ndarray:
        """Run a two-stem model and return its target stem (the caller derives the
        complement as x - target, so the pair always sums back to x exactly)."""
        out = self._run_model(model_file, x, job, overlap, shifts, complement=False)
        for n in names:
            if n in out:
                return out[n]
        if "instrumental" in out:
            return x - out["instrumental"]
        raise RuntimeError(f"{model_file}: unexpected stems {sorted(out)}")

    def _separate(self, job: Job, preset: Preset, mix44: np.ndarray, prog) -> dict[str, np.ndarray]:
        overlap, shifts = QUALITY[job.quality]

        if preset.key == "sw6":
            self._stage(prog, "6 stems (BS-RoFormer SW)")
            s = self._run_model(SW_6STEM, mix44, job, overlap, shifts)
            return {k: s[k] for k in preset.stems}

        if preset.key == "max6":
            self._stage(prog, f"vocals ({VOCAL_MODELS.get(job.vocal_model, job.vocal_model)})")
            vocals = self._target(job.vocal_model, mix44, job, overlap, shifts, ("vocals",))
            instrumental = mix44 - vocals          # exact complement
            self._stage(prog, "instruments (BS-RoFormer SW)")
            s = self._run_model(SW_6STEM, instrumental, job, overlap, shifts)
            # Whatever SW still calls "vocals" inside the instrumental is bleed/FX - keep it in "other"
            return {"vocals": vocals, "drums": s["drums"], "bass": s["bass"],
                    "guitar": s["guitar"], "piano": s["piano"], "other": s["other"] + s["vocals"]}

        if preset.key == "demucs4":
            self._stage(prog, "4 stems (HTDemucs FT)")
            s = self._run_model(DEMUCS_FT, mix44, job, overlap, shifts)
            return {k: s[k] for k in preset.stems}

        if preset.key == "vox2":
            self._stage(prog, "vocals")
            vocals = self._target(job.vocal_model, mix44, job, overlap, shifts, ("vocals",))
            return {"vocals": vocals, "instrumental": mix44 - vocals}

        raise ValueError(preset.key)

    def _get_separator(self, job: Job, overlap: int, shifts: int):
        ensure_ffmpeg()
        _patch_audio_separator()
        from audio_separator.separator import Separator

        if not job.use_directml:
            # AMD / Intel GPUs on Windows: use DirectML automatically when installed and no CUDA.
            try:
                import torch
                if not torch.cuda.is_available():
                    import torch_directml  # noqa: F401
                    job.use_directml = True
            except ImportError:
                pass
        key = (overlap, shifts, job.use_directml)
        if self._sep is None or self._sep_key != key:
            handler_level = logging.INFO
            self._sep = Separator(
                log_level=handler_level,
                model_file_dir=str(models_dir()),
                output_dir=tempfile.gettempdir(),
                output_format="WAV",
                normalization_threshold=1.0,
                amplification_threshold=0.0,
                sample_rate=MODEL_SR,
                use_soundfile=True,
                use_autocast=False,          # full fp32 inference for best quality
                use_directml=job.use_directml,
                mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                             "batch_size": 1, "overlap": overlap, "pitch_shift": 0},
                demucs_params={"segment_size": "Default", "shifts": shifts,
                               "overlap": 0.25, "segments_enabled": True},
            )
            self._sep_key = key
            self._loaded_key = None
        return self._sep

    def _run_model(self, model_file: str, mix44: np.ndarray, job: Job,
                   overlap: int, shifts: int, complement: bool = True) -> dict[str, np.ndarray]:
        self._check_cancel()
        sep = self._get_separator(job, overlap, shifts)
        if self._loaded_key != model_file:
            self._free_gpu()
            self._emit(f"Loading model {model_file} (downloads on first use)...")
            sep.load_model(model_filename=model_file)
            self._loaded_key = model_file

        with tempfile.TemporaryDirectory(prefix="stemsplit_") as td:
            tmp = Path(td) / "mix.wav"
            sf.write(str(tmp), mix44.T, MODEL_SR, subtype="FLOAT")
            _capture.clear()
            sep.separate(str(tmp))
            out = {k: v.T.astype(np.float32) for k, v in _capture.items()}
            _capture.clear()

        n = mix44.shape[1]
        out = {k: fit_length(v, n) for k, v in out.items()}
        # normalise names that differ between models
        if "vocals" not in out:
            for alt in ("vocal", "voice"):
                if alt in out:
                    out["vocals"] = out.pop(alt)
        if complement and "vocals" not in out and "instrumental" in out:
            out["vocals"] = mix44 - out["instrumental"]
        self._emit(f"{model_file}: got stems {sorted(out)}")
        return out

    @staticmethod
    def _free_gpu():
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _emit(self, msg: str):
        log.info(msg)
        if self._log_cb:
            self._log_cb(msg)


# --------------------------------------------------------------------------------------
# CLI:  python -m stemsplitter.engine song.mp3 [-p max6] [-o outdir]
# --------------------------------------------------------------------------------------

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Split a song into stems")
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("-p", "--preset", choices=list(PRESETS), default="max6")
    ap.add_argument("-q", "--quality", choices=list(QUALITY), default="High")
    ap.add_argument("-f", "--format", choices=list(OUTPUT_FORMATS), default="WAV 32-bit float")
    ap.add_argument("--vocal-model", default=DEFAULT_VOCAL_MODEL)
    ap.add_argument("--residual", choices=["none", "file", "other"], default="none")
    ap.add_argument("--split-vocals", action="store_true", help="lead / backing vocals")
    ap.add_argument("--dereverb", action="store_true", help="dry + reverb vocal stems")
    ap.add_argument("--split-drums", action="store_true", help="kick, snare, toms, hi-hat, ride, crash")
    ap.add_argument("--no-full", action="store_true", help="don't also write the stems that were split")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import glob
    inputs: list[Path] = []
    for p in a.inputs:   # cmd.exe does not expand wildcards
        hits = sorted(glob.glob(str(p)))
        inputs += [Path(h) for h in hits] if hits else [p]
    eng = Engine()
    print("Device:", eng.device_info())
    for inp in inputs:
        out = (a.output or inp.parent / "stems") / inp.stem
        job = Job(inp, out, a.preset, a.vocal_model, a.quality, a.format, a.residual,
                  split_vocals=a.split_vocals, dereverb=a.dereverb, split_drums=a.split_drums,
                  keep_full=not a.no_full)
        res = eng.run(job, lambda f, m: print(f"[{f*100:5.1f}%] {m}"))
        print(f"Residual (unassigned) energy: {res.residual_db:.1f} dB")
        for k, p in res.stems.items():
            tag = " (full, also split)" if k in res.full_stems else ""
            print(f"  {k:22s} {p}{tag}")


if __name__ == "__main__":
    sys.exit(main())
