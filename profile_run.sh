#!/usr/bin/env bash
# profile_run.sh [--fence] [--fp32] [-o OUTDIR] TRACK... : timing profile run.
# Samples the GPU at 100 ms while stemify --json --profile --force renders into
# OUTDIR (scratch), then prints the summary. --fence = low-priority systemd fence (see README).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$HERE/.venv/bin/python3
fence=0; extra=(); out=profile-runs/run-$(date +%Y%m%d-%H%M%S)
while [ $# -gt 0 ]; do
    case $1 in
        --fence) fence=1 ;;
        --fp32) extra+=(--fp32) ;;
        --compile) extra+=(--compile) ;;
        --aot) extra+=(--aot) ;;
        --compile-mode) extra+=(--compile --compile-mode "$2"); shift ;;
        --keep-stems) extra+=(--keep-stems) ;;
        -o) out=$2; shift ;;
        *) break ;;
    esac
    shift
done
[ $# -gt 0 ] || { echo "usage: $0 [--fence] [--fp32] [-o OUTDIR] TRACK..." >&2; exit 2; }
mkdir -p "$out"
cmd=("$PY" "$HERE/stemify" --json --profile --force -o "$out/stems" "${extra[@]}" "$@")
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100 \
    > "$out/gpu.csv" & smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
if [ $fence = 1 ]; then
    systemd-run --user --wait --pipe --collect --unit="stemify-profile-$$" \
        --working-directory="$HERE" \
        -p Nice=19 -p CPUWeight=20 -p IOSchedulingClass=idle \
        ${STEMIFY_ALLOWED_CPUS:+-p AllowedCPUs=$STEMIFY_ALLOWED_CPUS} -E OMP_NUM_THREADS=4 \
        "${cmd[@]}" > "$out/events.jsonl" 2> "$out/stemify.err"
else
    "${cmd[@]}" > "$out/events.jsonl" 2> "$out/stemify.err"
fi
kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true
trap - EXIT
"$PY" "$HERE/profile_summary.py" "$out"
echo "run dir: $out"
