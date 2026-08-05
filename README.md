# Stems for Mixxx

Batch-renders ordinary tracks into Native Instruments `.stem.mp4` files so
Mixxx 2.6+ can drive drums / bass / other / vocals as independent deck faders.

This is **not** what djay does. djay separates in real time on whatever you drop
on the deck. Mixxx plays *pre-rendered* stem files, so the library has to be
prepared in advance.

## Usage

```bash
stemify track.flac -o ~/Music/stems          # one track
stemify ~/Music/dj/*.flac -o ~/Music/stems   # batch, model loads once
stemify --keep-stems track.flac              # also keep the raw wavs
```

Already-rendered tracks are skipped unless `--force` is passed, so re-running
over a folder only does the new arrivals.

## How it works

1. **Separate** — `htdemucs_ft` on the GPU. Best 4-stem SDR of the models
   available (vocals 10.8, drums 10.0, bass 12.0). Roughly 3.5x realtime on the
   RTX 3060; a 4-minute track takes about 75 s.
2. **Mux** — ffmpeg writes 5 AAC streams: the untouched original as stream 0
   (the master mix), then the four stems. Mixxx requires exactly 5 streams with
   identical codec and sample rate.
3. **Tag** — the NI manifest goes in as a `moov.udta.stem` atom. Mixxx probes
   for that atom and falls back to the `.stem.mp4` extension; the atom is what
   supplies the stem labels and colours.

The atom is appended after muxing, which is only safe because ffmpeg writes
`moov` *after* `mdat`. Growing a leading `moov` would shift `mdat` and
invalidate every `stco` chunk offset. `inject_manifest()` refuses to write if
`moov` is not the last box.

## Layout

- `stemify` — the tool (`~/bin/stemify` is a thin wrapper)
- `.venv/` — Python 3.12 + torch cu128 + audio-separator. Separate from system
  Python, which is 3.14 and too new for torch.
- `models/` — cached model weights, ~322 MB. The library otherwise defaults to
  `/tmp`, which on endgame means re-downloading after every reboot.

## Caveats

- Stem files are big: a 4-minute track lands around 40 MB at 256k, since it is
  five audio streams rather than one.
- Separation is lossy. Reverb tails and heavily-processed material smear between
  stems; percussive and well-separated mixes come out best. Check anything
  before you play it out.
- Mixxx needs to be built with `STEM` enabled. Arch's `extra/mixxx` is not —
  `mixxx-beta` from the AUR is, since the flag auto-enables when ffmpeg is found.
