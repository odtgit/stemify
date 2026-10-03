# Host integration notes

Open items for driving stemify from a DJ app (first target: a Flatpak Mixxx
fork) as a background stem queue. Not implemented yet.

## Process model

- **Resident worker.** Keep both models loaded (AOT packages: ~11 s startup,
  ~6 GB VRAM) and take jobs over a socket or job directory, instead of one
  process per batch. Startup is small now, but a worker also keeps CUDA
  context and pinned buffers warm and makes cancellation/priority possible.
- **Flatpak boundary.** stemify stays on the host (CUDA wheels, 1.6 GB models,
  1.6 GB AOT packages). App side talks to it via `flatpak-spawn --host` or a
  host service over a socket/D-Bus.
- **Event contract.** Today: `--json` line events (`start`, `track_start`,
  `track_done`, `track_failed`, `done`), stderr notices, exit codes. A worker
  needs job ids, progress within a track (per pass / per chunk), cancel, and a
  capabilities query (device, VRAM, backend, models present).

## Latency (single track matters more than batch throughput)

- **Chunk batching.** Run 2-4 chunks of the same track per forward pass
  (`inference.batch_size`). GPU is ~95% busy over time but batch-1 kernels
  likely leave SMs idle, so this could cut single-track wall time. Needs: AOT
  packages per batch size, harness run for speed + stem diff, VRAM check
  (+1.5-2 GB expected). Two parallel processes are not the answer: ~2x VRAM
  (does not fit 12 GB with a desktop) and contexts time-slice rather than
  overlap.
- **Decode prefetch.** Decode track N+1 on the post worker while the GPU is
  still on N. Removes the ~1-1.6 s idle gap between tracks seen in batches
  (GPU idle ~5%).
- Current numbers for reference: ~11 s startup + ~45 s per 4-minute track
  (AOT, RTX 3060).

## App-side policy

- **Triggers:** manual, on import, on crate/playlist add, on deck load.
- **Busy rules:** pause while recording, or while any deck plays; GPU
  contention with waveform rendering has no priority knob on consumer NVIDIA.
- **Source to stem link.** A `.stem.mp4` is a separate file, so the library
  needs to map source track to stem file and carry cues, loops and beatgrid
  across (or swap the track's location). Biggest design item.
- After a render the app must re-read tags; Mixxx does not rescan changed
  files.

## Distribution

- Models are not bundled; download on first use with consent and size shown,
  verify sha256.
- BS-RoFormer SW weights have unclear provenance: opt-in only in any public
  build; default to clean-licence models.
- Presets by hardware: eager/AOT on NVIDIA; other devices untested.
