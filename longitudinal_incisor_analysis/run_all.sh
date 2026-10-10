#!/usr/bin/env bash
#
# Run enamel_density.py over every sample that has a CMPR resliced volume.
#
# Each sample is paired with its molar, which supplies the flat dentin baseline
# (--baseline molar is the default). Without -m the run falls back to the
# per-slice core trace, which inflates the dentin reference toward the tip.
#
# Samples are DISCOVERED from the CMPR directory rather than hardcoded. An
# earlier version carried a fixed list, and the Sep-2026 rename
# (Odam-CRE-A -> P52_Odam-Cre_WT_1, and so on; see sample_rename_map.csv)
# invalidated every entry. Because the script skips a missing volume rather
# than failing, that silently turned the whole run into a no-op that still
# exited 0. Globbing cannot go stale that way, and it picks up new samples --
# the P82 CFTR-cKO series was never in the fixed list at all.
#
# Usage:  ./run_all.sh [output_dir] [-- extra enamel_density.py args]

set -uo pipefail

BASE="/Volumes/Z Drive/Incisor Enamel Analysis"
OUT="${1:-$BASE/RESULTS}"
shift || true
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

cd "$(dirname "$0")"

if [[ ! -d "$BASE/CMPR" ]]; then
  echo "ERROR: $BASE/CMPR is not reachable. Is the Z drive mounted?" >&2
  exit 1
fi

# Derive sample names from the resliced volumes on disk. The ._* entries are
# macOS AppleDouble sidecars on this exFAT volume and are not samples.
SAMPLES=()
for v in "$BASE/CMPR"/*_resliced_volume.nrrd; do
  [[ -f "$v" ]] || continue
  name="$(basename "$v" _resliced_volume.nrrd)"
  [[ "$name" == ._* ]] && continue
  SAMPLES+=("$name")
done

if ((${#SAMPLES[@]} == 0)); then
  echo "ERROR: no *_resliced_volume.nrrd found in $BASE/CMPR" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "Output:  $OUT"
echo "Samples: ${#SAMPLES[@]}"
((${#EXTRA[@]:-0})) && echo "Extra:   ${EXTRA[*]:-}"
echo

ok=0
failed=()
for s in "${SAMPLES[@]}"; do
  incisor="$BASE/CMPR/${s}_resliced_volume.nrrd"
  molar="$BASE/SEG DATA/$s/volumes/${s}_molar_volume.nrrd"

  if [[ ! -f "$molar" ]]; then
    echo "SKIP $s -- no molar volume; would fall back to the traced reference"
    failed+=("$s (no molar)")
    continue
  fi

  echo "=== $s"
  # "${EXTRA[@]+"${EXTRA[@]}"}" rather than "${EXTRA[@]}": macOS ships bash
  # 3.2, where expanding an EMPTY array under `set -u` is an unbound-variable
  # error. The +alternate form expands to nothing when the array is unset or
  # empty, and to the properly quoted elements otherwise.
  if python enamel_density.py -i "$incisor" -m "$molar" -o "$OUT" -l \
       "${EXTRA[@]+"${EXTRA[@]}"}"; then
    echo "--- $s done"
    ((ok++))
  else
    echo "--- $s FAILED"
    failed+=("$s (run error)")
  fi
  echo
done

echo "=================================================================="
echo "Analyzed $ok of ${#SAMPLES[@]} samples."
if ((${#failed[@]:-0})); then
  printf 'Not analyzed:\n'
  printf '  - %s\n' "${failed[@]+"${failed[@]}"}"
fi
# A run that analyzed nothing is a failure, not a success.
((ok > 0)) || exit 1
