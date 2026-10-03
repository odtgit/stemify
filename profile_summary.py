#!/usr/bin/env python3
"""Summarise a stemify --profile run: join gpu.csv samples to stage windows.

usage: profile_summary.py RUNDIR [RUNDIR2 ...]
RUNDIR holds events.jsonl (stemify --json --profile) and gpu.csv
(nvidia-smi timestamp,util,mem). Several RUNDIRs also get a hash comparison.
"""
import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore", RuntimeWarning)

STAGES = ["decode", "vocal_pass", "inst", "sw_pass", "residual",
          "write_wavs", "mux", "finalize"]


def load_gpu(p):
    ts, util, mem = [], [], []
    for line in Path(p).read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        try:
            ts.append(datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp())
            util.append(float(f[1]))
            mem.append(float(f[2]))
        except ValueError:
            continue
    return np.array(ts), np.array(util), np.array(mem)


def load(d):
    d = Path(d)
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()
          if l.startswith("{")]
    return (ev, *load_gpu(d / "gpu.csv"))


def gpu_mean(g, t0, t1):
    ts, util, _ = g
    m = (ts >= t0) & (ts <= t1)
    return float(util[m].mean()) if m.any() else float("nan")


def summarise(d):
    ev, *g = load(d)
    g = tuple(g)
    proc = next((e for e in ev if e["event"] == "profile_process"), None)
    tracks = [e for e in ev if e["event"] == "track_done" and "profile" in e]
    print(f"\n## {d}  ({len(tracks)} tracks)")
    if proc:
        s, m = proc["startup"], proc["model_load"]
        print(f"startup {s['wall']:.1f}s (cpu {s['cpu']:.1f}s)  model_load "
              f"{m['wall']:.1f}s (cpu {m['cpu']:.1f}s, gpu {gpu_mean(g, m['t0'], m['t1']):.0f}%)"
              f"  fp32={proc.get('fp32')}")
    print("\nper track (s): audio  total  " + "  ".join(STAGES) + "  other_extra_db")
    for t in tracks:
        p = t["profile"]
        print(f"{Path(t['src']).name[:40]:40s} {t['audio_secs']:6.1f} {t['secs']:6.1f}  "
              + "  ".join(f"{p[k]['wall']:.1f}" for k in STAGES)
              + f"  {t['other_extra_db']}")
    if not tracks:
        return None
    tot = np.array([[t["profile"][k]["wall"] for k in STAGES] for t in tracks])
    cpu = np.array([[t["profile"][k]["cpu"] if t["profile"][k]["cpu"] is not None
                     else np.nan for k in STAGES] for t in tracks], float)
    gu = np.array([[gpu_mean(g, t["profile"][k]["t0"], t["profile"][k]["t1"])
                    for k in STAGES] for t in tracks])
    mins = np.array([t["audio_secs"] for t in tracks]) / 60
    track_wall = np.array([t["secs"] for t in tracks])
    print(f"\n{'stage':11s} {'wall s':>7s} {'%track':>7s} {'gpu %':>6s} {'cores':>6s}"
          f"   fit wall = a + b*min (a s, b s/min)")
    for i, k in enumerate(STAGES):
        fit = ""
        if len(tracks) >= 2:
            b, a = np.polyfit(mins, tot[:, i], 1)
            fit = f"a={a:6.2f}  b={b:6.2f}"
        print(f"{k:11s} {tot[:, i].mean():7.2f} {100 * (tot[:, i] / track_wall).mean():7.1f} "
              f"{np.nanmean(gu[:, i]):6.0f} {(np.nansum(cpu[:, i]) / tot[:, i].sum()):6.2f}   {fit}")
    io = tot[:, STAGES.index("write_wavs"):].sum(1)
    print(f"write_wavs+mux+finalize: {100 * (io / track_wall).mean():.1f}% of wall; "
          f"track total mean {track_wall.mean():.1f}s, "
          f"realtime x{(mins * 60 / track_wall).mean():.2f}")
    done = next((e for e in ev if e["event"] == "done"), {})
    if "batch_wall" in done:
        print(f"batch_wall {done['batch_wall']:.1f}s vs sum of track walls "
              f"{track_wall.sum():.1f}s (post stages overlap next track's passes: cpu n/a)")
    return {Path(t["src"]).name: t["profile"]["stems"] for t in tracks}


def main():
    res = [summarise(d) for d in sys.argv[1:]]
    if len(res) > 1 and all(res):
        base = res[0]
        print("\n## stem hash / RMS comparison vs", sys.argv[1])
        for d, r in zip(sys.argv[2:], res[1:]):
            for name, st in r.items():
                b = base.get(name)
                if not b:
                    continue
                same = {k: st[k]["sha256"] == b[k]["sha256"] for k in st}
                dr = {k: round(st[k]["rms_db"] - b[k]["rms_db"], 3) for k in st}
                print(f"{d} {name[:36]}: identical={same}  rms_db delta={dr}")
    for d, r in zip(sys.argv[1:], res):
        if r:
            print(f"\n## stem checksums {d}")
            for name, st in r.items():
                print(name[:40], " ".join(f"{k}={v['sha256'][:12]}({v['rms_db']:.2f}dB)"
                                          for k, v in st.items()))


if __name__ == "__main__":
    main()
