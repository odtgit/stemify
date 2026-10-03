# stemify timing baseline (2026-10-02)

## Machine
RTX 3060 12 GB, driver 615.71.09 (CUDA 12.8 build), AMD Ryzen 9 5950X (16C/32T), torch 2.11.0+cu128, msst 0.1.0, fp16 autocast (MSST default), vocal overlap 2, SW overlap per YAML. Desktop was running (GPU idle-to-20% from the compositor, no other renders active).

## Method
`./profile_run.sh [--fence] [--fp32] -o DIR tracks...`: `stemify --json --profile --force`, CUDA synced at every stage boundary, nvidia-smi sampled every 100 ms and joined to stage windows by epoch time. Wall per stage, CPU = process + children CPU delta, cores = CPU/wall. Fence = low-priority batch properties (Nice 19, CPUWeight 20, idle IO, AllowedCPUs 2-15,18-31 via STEMIFY_ALLOWED_CPUS, OMP_NUM_THREADS=4) via `systemd-run --user --wait --pipe`. Tracks: 2:30, 3:41, 4:01, 5:24 (one process per run, so all four share one model load).

## Per-track wall (s), unfenced run 1 / run 2 / fenced
| track | audio | unfenced-1 | unfenced-2 | fenced |
|---|---|---|---|---|
| Ain't No Mountain | 149.5 | 42.9 | 42.6 | 42.9 |
| Give Me the Night | 221.2 | 68.5 | 61.7 | 64.6 |
| Re-Rewind | 240.7 | 69.0 | 68.1 | 69.6 |
| Rain | 323.8 | 85.9 | 87.3 | 89.4 |
| mean (x realtime) | | 66.6 (3.49) | 64.9 (3.58) | 66.6 (3.50) |

Run 1 Give Me the Night is a +7 s outlier on the GPU stages (22.4 s vs 24.2 s vocal, 29.8 s vs 34.6 s SW), no cause found.

## Per-stage means (4 tracks)
| stage | unf-1 wall s | unf-2 wall s | fenced wall s | % of track (unf-2) | GPU % (unf-2 / fenced) | cores (unf-2 / fenced) |
|---|---|---|---|---|---|---|
| decode | 0.09 | 0.09 | 0.09 | 0.1 | 5 / 1 | 3.8 / 3.9 |
| vocal_pass | 23.93 | 23.53 | 24.18 | 36.4 | 97 / 97 | 1.20 / 1.04 |
| inst | 0.01 | 0.01 | 0.01 | 0.0 | n/a | 1.0 |
| sw_pass | 32.64 | 31.47 | 32.33 | 48.4 | 96 / 96 | 1.35 / 1.06 |
| residual | 0.17 | 0.17 | 0.16 | 0.3 | 0 / 4 | 17.6 / 2.7 |
| write_wavs | 0.32 | 0.32 | 0.32 | 0.5 | 1 / 3 | 0.55 / 0.56 |
| mux | 8.94 | 8.87 | 9.03 | 13.6 | 2 / 10 | 2.73 / 2.75 |
| finalize | 0.10 | 0.10 | 0.10 | 0.2 | 5 / 5 | 0.86 / 0.85 |

write_wavs+mux+finalize: 14.2% of wall (unfenced), 14.1% (fenced).
Mux is ~2.1 s per audio minute (AAC encode of 5 streams, ~2.7 cores, no GPU). The GPU is idle for it.

## Linear fit, wall = a + b * audio_minutes (unfenced-2; s, s/min)
| stage | a | b |
|---|---|---|
| vocal_pass | 2.49 | 5.40 |
| sw_pass | 1.71 | 7.64 |
| mux | 0.76 | 2.08 |
| write_wavs | -0.02 | 0.09 |
| other stages | ~0 | <0.05 |

Total ~ 15.3 s per audio minute (3.9 min track ~ 62 s once the model is loaded).

## Process
startup 0.9 s (torch+msst import, warm cache), model_load 3.4-3.7 s (both models). Total 4.5 s per process, same fenced.

## Reproducibility baseline
Stem sha256 (float32, before AAC) identical across unfenced-1, unfenced-2 and fenced for all four tracks. Hashes are in each run's summary (`profile_summary.py DIR`; pass several dirs to diff). Other-extra dB: -22.24, -23.21, -21.18, -27.63.

## fp32 (Give Me the Night, once)
| | fp16 (unfenced-2) | fp32 |
|---|---|---|
| track wall | 61.7 s | 140.1 s (2.3x) |
| vocal_pass / sw_pass | 22.4 / 29.8 s | 55.6 / 74.8 s |
| other-extra | -23.21 dB | -23.2 dB |
| stem RMS dB (d/b/o/v) | -26.43 / -29.75 / -28.92 / -27.00 | same to 0.01 dB |

Hashes differ, as expected. Caveat: the SW model's attention is flash-only on this GPU, which cannot run fp32. `--fp32` switches both models to math/mem-efficient kernels, so the timing includes that change. No audible-quality conclusion drawn, only level and other-extra match.

## Decision rows
| rule | measured | triggered |
|---|---|---|
| write_wavs+mux+finalize > 10% of wall | 14.1% | YES: overlap mux/writes with the next track's GPU work (saves up to ~9 s of ~65 s per track, ~14%) |
| GPU util < ~85% in vocal/sw pass | 96-97% | no |
| GPU ~100% in passes | 96-97% | YES: only kernel-level help (torch.compile) speeds the passes. nvidia-smi util is "any kernel running" time, not SM occupancy, so batch_size > 1 is not ruled out, only unsupported by this metric |
| startup + model_load > ~15 s | 4.5 s | no |
| fence slows GPU stages | vocal +2.8%, sw +2.7% vs unf-2 (unfenced run-to-run spread is +1.7% vocal, +3.7% sw); track mean +2.6% | no |

## Notes
- Fence cost here is near zero because the box was idle; cost under CPU load is not tested.
- Fenced residual uses 2.7 cores vs 17.6 unfenced (OMP cap), but the stage is 0.17 s.

## Mux overlap (2026-10-03)
write_wavs + mux + finalize run in one background worker while the next track is separated; at most one track is pending (bounded memory), scratch is per track. Same four tracks, one process, `overlap-1`.

| | baseline (unfenced-2) | overlap |
|---|---|---|
| batch wall | 259.8 s (sum of track walls) | 234.3 s (-9.8%) |
| vocal_pass mean | 23.53 s | 23.45 s |
| sw_pass mean | 31.47 s | 31.59 s |
| per-track mux wall | 8.87 s | 8.86 s |
| stem sha256 | | identical for all 4 tracks x 4 stems |

ffmpeg's ~2.7 cores do not slow the GPU passes (within run-to-run spread). Saving is the first three tracks' post stages (~25 s); the last track's ~11 s post stage is not hidden. Per-track `secs` still spans separation to finalize; post-stage walls are measured in the worker without a CUDA sync (cpu is null, not attributable). `track_start` N+1 precedes `track_done` N. Failure path (bogus input mid-batch): `track_failed` in order, batch continues, exit 1, no partial output.

## torch.compile (2026-10-03)
`--compile` calls `nn.Module.compile()` on the module MSST holds (`Separator._model`; `demix` calls `model(arr)`, so the compiled forward is the one used) and warms it with one silent chunk per model. Chunks are always padded to chunk_size and both inference batch sizes are 1, so shapes are fixed: no recompiles seen. Same four tracks, one process, `compile-1` (default mode), keep-stems eager run `keep-eager` for the diff.

| | eager+overlap | compile default | compile max-autotune |
|---|---|---|---|
| warm-up (cold cache) | 0 | 224.9 s | 424.9 s |
| warm-up (warm inductor cache) | 0 | 72.2 s | n/a |
| vocal_pass mean | 23.45 s | 13.62 s (1.72x) | 12.81 s (1.83x) |
| sw_pass mean | 31.59 s | 17.81 s (1.77x) | 16.83 s (1.88x) |
| batch wall excl. warm-up | 234.3 s | 140.1 s | 132.7 s |
| batch wall incl. cold warm-up | 234.3 s | 365.0 s | 557.6 s |
| steady s per track (GPU bound, mux hidden) | 55.0 | 31.4 | 29.6 |

Break-even vs eager: 225 / (55.0 - 31.4) = ~9.5 tracks of this mix (~3.2 min average) cold, ~3 tracks with a warm cache. max-autotune gains 6% over default for 200 s more warm-up.

Per-stem difference vs eager float stems, 10log10(E[diff]/E[ref]) dB (default mode):

| track | drums | bass | other | vocals |
|---|---|---|---|---|
| Ain't No Mountain | -61.9 | -53.5 | -54.8 | -75.0 |
| Give Me the Night | -69.0 | -63.3 | -63.8 | -74.7 |
| Re-Rewind | -59.9 | -71.5 | -53.8 | -71.3 |
| Rain | -63.5 | -66.7 | -59.5 | -72.2 |

Other-extra unchanged to 0.05 dB (-22.24, -23.21, -21.18, -27.63). Hashes differ, as expected. Worst stem -53.5 dB: below the -50 dB opt-in bar, short of the -60 dB default bar. With compile the post stage (~9 s) is still hidden behind ~31 s of GPU work per track.

### Persistent compile cache
`--compile` defaults `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to `${XDG_CACHE_HOME:-~/.cache}/stemify/{inductor,triton}` (unless already set), before torch is imported; previously /tmp (tmpfs, cold after every reboot). `fx_graph_cache` and `enable_autograd_cache` are already True in torch 2.11. One track (Marvin Gaye), empty cache dir then reuse: warm-up 219.0 s cold, 72.0 s warm; cache is 184 MB inductor + 132 MB triton. Runs `cache-cold`, `cache-warm`.

Mega-cache (`torch.compiler.save_cache_artifacts`/`load_cache_artifacts`): saved after a warm run (166 MB, 4 FX graphs plus autotune configs), loaded into an empty cache dir at startup in 0.3 s: warm-up 216.1 s, i.e. no gain over a cold start (the 72 s warm figure needs the on-disk cache entries; the artifact does not reproduce them). Not added.

## AOTInductor (`--aot`, 2026-10-03)
Each model's forward is split: STFT, band indexing, complex mask apply and iSTFT stay eager; band split + transformers + mask heads (`Core`) are `torch.export`ed (strict=False, batch 1, 2 ch, YAML chunk_size; both inference batch sizes are 1) under autocast and packaged with `aoti_compile_and_package`, then loaded with `aoti_load_package` in place of `band_split` (layers emptied, heads replaced by index modules, so MSST's `demix` is untouched).
Obstacles hit and fixes: (1) whole-model export fails in AOT codegen (`aten.full`/`select` fallbacks from the complex/STFT section), hence the split; (2) rotary `cached_freqs` buffers are mutated on first call ("duplicated inputs that are mutated"): one eager chunk first fills the cache; (3) autocast casts are not captured for sdpa (dtype assert): sdpa is made explicit fp16 (identical to autocast's policy); (4) AOTI compile needs a CUDA toolkit (`CUDA_HOME`: cuda_runtime.h, crt/, nv/target, libcudart.so); the box has none; the first test used a throwaway shim (triton include tree + cccl headers + libcudart symlink), replaced by the wheel toolkit below.

Marvin Gaye, empty cache dir then reuse: build 178 s (first run, incl. inductor compile), reuse: model_load 3.7 s + warm-up 7.7 s (eager chunk for shape/rotary cache plus one silent chunk each) = ~11.4 s. Cache 1.6 GB .pt2 (fp32 constants inside) + 1.6 GB inductor. Four tracks, one process (`aot-4`, diff vs `keep-eager`):

| | eager+overlap | compile default | aot |
|---|---|---|---|
| vocal_pass mean | 23.45 s | 13.62 s | 12.94 s |
| sw_pass mean | 31.59 s | 17.81 s | 16.70 s |
| warm-up (warm cache) | 0 | 72.2 s | 7.7 s |
| worst stem diff | | -53.5 dB | -53.6 dB |
| peak VRAM (nvidia-smi) | 4.97 GB | 4.79 GB | 5.89 GB |

Per-stem diff is the same as compile to 0.1-1 dB (aot: Marvin bass -53.6, other -54.8; Artful other -53.8; others -59.7 to -75.1). Meets the bars (startup < 15 s, passes within 10% of compile, worst < -50 dB). Cost: needs a CUDA toolkit for the one-off build, 1.6 GB cache, +1.1 GB VRAM. Superseded: AOT is now the default, see the last section.

Recommendation: keep opt-in. Steady-state speedup is 1.75x, but warm-up is 225 s cold / 72 s warm (limit ~60 s) and the worst stem difference is -53.5 dB.

## AOT as default, wheel toolkit (2026-10-03)
Default backend is now AOT (package exists: load; else toolkit available: build; else eager with a stderr notice; `--aot` forces, `--eager`/`--fp32` bypass). The build toolkit is three version-matched wheels instead of a system CUDA: `nvidia-cuda-runtime-cu12==12.8.90` (headers, libcudart.so.12; already a torch dependency), `nvidia-cuda-nvcc-cu12==12.8.93` (crt/ headers), `nvidia-cuda-cccl-cu12==12.8.90` (cuda/, cub/, nv/). stemify symlinks their include dirs plus `lib64/libcudart.so` into `${XDG_CACHE_HOME:-~/.cache}/stemify/cudahome` and sets `CUDA_HOME` to it when unset; no compiler from the wheels is used (the host g++ builds the package).

Empty cache, CUDA_HOME unset, Marvin Gaye: build warm-up 178.0 s (startup 1.0 + model_load 5.8 s, then 25.8 s track); reuse: startup 1.1 + model_load 4.2 + warm-up 6.1 = 11.4 s. Four tracks, one process (`default-4`): vocal_pass mean 13.10 s, sw_pass mean 16.80 s, batch wall 135.8 s (5.6x realtime). Per-stem diff vs `aot-4`: worst -55.5 dB (others -61 to -76); vs `keep-eager`: worst -53.3 dB (Marvin bass), others -54.5 to -75.8. Cache: 1.6 GB aot packages, 1.6 GB inductor intermediates (deletable after build), cudahome symlinks only.
