# Stem Splitter

Free, open-source Windows app that splits songs into stems for music production and remixing: vocals, drums, bass, guitar, piano, other. It can also split further into lead and backing vocals, a dry vocal and its reverb, and the drum kit (kick, snare, toms, hi-hat, ride, crash).

It uses state-of-the-art open separation models (BS-RoFormer, Mel-RoFormer, MDX23C, HTDemucs) and writes **lossless, gain-accurate stems**: 32-bit float at the song's own sample rate, no normalisation, and stems that sum back to the original mix.

## Download

1. Download **`StemSplitter-vX.Y.Z-windows.zip`** from the [latest release](https://github.com/hmcelik/stem-splitter/releases/latest).
2. Extract it anywhere, for example `D:\StemSplitter`.
3. Double-click **`run.bat`**.

The first start installs everything automatically: Python if it's missing, then PyTorch (the GPU build if you have an NVIDIA card) and the other packages. That's about 3–4 GB and takes a while, but only happens once. Each separation model downloads the first time it's used (0.2–0.9 GB each).

**Requirements:** Windows 10/11 64-bit. An **NVIDIA GPU** is strongly recommended. It works on CPU, but slowly.

> Why not a single `.exe`? The GPU build of PyTorch alone is larger than GitHub's 2 GB limit per release file. The installer approach also keeps the download small and lets anyone edit the code.

## Features

- Drag songs (WAV, FLAC, MP3, M4A, AIFF, OGG…) or whole folders into a queue.
- Each stem gets **Play**, **Save as…** (WAV 32f / WAV 24 / FLAC 24), **Show**, or **drag the row straight into your DAW**.
- Batch processing in the background, with settings remembered between sessions.
- Command-line mode for scripting.

## Separation modes

| Mode | Models | Stems |
|---|---|---|
| **6 stems – Maximum quality** (default) | dedicated vocal RoFormer → BS-RoFormer SW on the instrumental | vocals, drums, bass, guitar, piano, other |
| 6 stems – BS-RoFormer SW | BS-RoFormer SW (one pass, about 2× faster) | same six |
| 4 stems – HTDemucs FT | Demucs v4 fine-tuned (lower VRAM) | vocals, drums, bass, other |
| 2 stems – Vocals / Instrumental | vocal RoFormer | vocals, instrumental |

The vocal model is selectable: BS-RoFormer 1297, BS-RoFormer Resurrection, Mel-RoFormer becruily, or Mel-RoFormer FV4.

### Extra splits (optional, any mode)

| Option | Splits | Into |
|---|---|---|
| Lead / backing vocals | vocals | `lead vocals`, `backing vocals` |
| Dry vocal + reverb | lead vocals (or vocals) | `… dry`, `… reverb` |
| Drum kit | drums | `kick`, `snare`, `toms`, `hi-hat`, `ride`, `crash` |

With everything on you get up to 13 stems. **Also keep the full stems** writes the original `vocals` / `drums` too. Kick and snare separate cleanly; cymbals bleed into each other more. Strings, brass/wind and synths have no public models yet, so they stay in `other`.

## Output quality

- **No normalisation.** Stems keep their real level relative to the song: faders at 0 dB rebuild the mix.
- **Original sample rate.** The models run at 44.1 kHz; stems are resampled back with SoX VHQ.
- **32-bit float by default.** No clipping, no dither.
- **Leftover.** What the models don't assign to any stem can be discarded, saved as a `residual` stem, or added to `other`. With either of the last two, the stems null against the original exactly (tested to −150 dBFS).

## Speed and settings

Speed depends mostly on the GPU, and especially on its **graphics memory (VRAM)**. When a model doesn't fit, Windows spills it into system RAM, which is many times slower.

| Setting | Options | Effect |
|---|---|---|
| **Precision** | Full (FP32) / Fast (FP16) | FP16 uses about half the VRAM and is usually much faster on NVIDIA GPUs, with a difference that's very hard to hear. Default for GPUs with ≤ 8 GB. |
| **Quality** | Fast / Standard / High / Extreme | Overlap between processing windows: 2 / 4 / 8 / 16. Each step doubles the time. |
| **Mode** | Maximum quality vs. BS-RoFormer SW | SW skips the separate vocal pass. |
| **Extra splits** | on / off | Each one adds a pass. |

Reference: an **RTX A1000 Laptop GPU (4 GB)** with an i7-13700H and 16 GB RAM processed a 4:35 song in *Maximum quality*, *Standard*, FP32, with all three extra splits, in about **24 minutes** (vocals 8:44, 6 stems 4:47, lead/backing 5:36, de-reverb 3:49, drum kit 1:12). On cards with 4–6 GB, use FP16. Measured FP16 numbers from users are welcome: open an issue with your GPU and timings.

## Run from source

```bat
git clone https://github.com/hmcelik/stem-splitter.git
cd stem-splitter
setup.bat
run.bat
```

`setup.bat` creates `.venv`, installs the requirements, and puts the CUDA build of PyTorch in last so nothing replaces it. Use `run_debug.bat` to see console output. The app also logs to `%LOCALAPPDATA%\StemSplitter\app.log`.

### Command line

```bat
.venv\Scripts\activate
python -m stemsplitter.engine "C:\Music\song.flac" -p max6 -q High --residual file -o "D:\Stems"
python -m stemsplitter.engine "C:\Music\album\*.flac" -p sw6 --split-vocals --split-drums --dereverb
```

### Project layout

```
app.py                    entry point (GUI)
stemsplitter/engine.py    separation pipeline, lossless audio I/O, CLI
stemsplitter/gui.py       PySide6 interface
setup.bat / run.bat       installer and launcher
.github/workflows/        CI check and release packaging
```

## Troubleshooting

- **`Device: CPU` despite an NVIDIA GPU:** update the GPU driver and run `setup.bat` again.
- **`No module named …`:** run `setup.bat` again. It only installs what's missing.
- **Very slow on an NVIDIA GPU / CUDA out of memory:** switch **Precision** to FP16, use *Fast* or *Standard* quality, and close other GPU-heavy apps (browsers and games use VRAM too).
- **AMD / Intel GPU:** run `.venv\Scripts\pip install torch-directml`. The app picks DirectML up automatically.
- **Model download fails:** check access to github.com and huggingface.co. Models are stored in `%LOCALAPPDATA%\StemSplitter\models`.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits and licenses

- Code: [MIT License](LICENSE).
- Separation runs on [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) (MIT) and the [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) model ecosystem.
- **Model weights are not included in this repository.** They're downloaded at runtime from their authors' releases, and each model has its own license, set by its authors (viperx, jarredou, aufr33, becruily, frazer, unwa, anvuew, Gabox, the Demucs authors, and others). Check a model's license before using its output commercially.
- Separating a song doesn't change who owns it. Only distribute stems of music you have the rights to.
