# Contributing

Thanks for helping out! Bug reports, model suggestions and pull requests are all welcome.

## Reporting a bug

Open an issue and include:
- what you did, and what happened instead
- the end of `%LOCALAPPDATA%\StemSplitter\app.log`, or the console output of `run_debug.bat`
- your GPU and Windows version

## Development setup

```bat
git clone https://github.com/hmcelik/stem-splitter.git
cd stem-splitter
setup.bat
.venv\Scripts\activate
python app.py
```

The engine can be tested without the GUI:

```bat
python -m stemsplitter.engine some_song.wav -p sw6 -q Standard --residual file
```

With `--residual file`, the leaf stems plus `residual` must sum back to the input exactly. Please keep that property when you change the pipeline.

## Guidelines

- Keep audio handling lossless: no normalisation, float32 internally, original sample rate on output.
- New models go through `audio-separator`, by filename. Add them to `engine.py` (presets or extra splits) and document them in the README.
- `audio-separator` is pinned. If you bump it, check that `_patch_audio_separator()` still works, since it relies on library internals.
- Keep pull requests focused, one feature or fix each.

## Releases

Releases are automatic. Bump `__version__` in `stemsplitter/__init__.py` (e.g. `0.2.0`) and push to `main`. `.github/workflows/release.yml` then checks that the app imports on Windows, packages `StemSplitter-v0.2.0-windows.zip`, and publishes the GitHub release with tag `v0.2.0`. Pushes that don't change the version don't create a release.
