#!/usr/bin/env bash
# Labyra QE Batch entrypoint — stage inputs from GCS, run QE on LOCAL scratch,
# stage selected outputs back to GCS. Driven by env vars (Batch sets them per task).
#
# Contract (set by the worker driver via batch runnable environment):
#   QE_BINARY    pw.x | bands.x | dos.x | projwfc.x | pp.x   (required)
#   QE_IN        input filename, e.g. pw_scf.in              (required)
#   QE_OUT       output filename (default: <QE_IN without .in>.out)
#   GCS_WORK     gs:// prefix for THIS unit (.in lives here; .out/out/ go here)  (required)
#   GCS_DEPS     space-separated gs:// prefixes whose out/ to restart from (scf→nscf/bands/…)
#   GCS_PSEUDO   gs:// prefix holding pseudopotential UPF files (optional)
#   NPROC        MPI ranks (default 1)
#   OMP_NUM_THREADS  OpenMP threads (default 1)
#
# The generated .in MUST set  outdir = './out'  (relative) so QE scratch lands in the
# local staged dir and the .save handoff round-trips through GCS.
#
# NOT set -e: we always upload the .out (even on QE failure, for diagnosis), then exit
# with QE's own code so Batch lifecyclePolicies (FAIL_TASK on 1/137/139) fire correctly.
set -uo pipefail

: "${QE_BINARY:?QE_BINARY required}"
: "${QE_IN:?QE_IN required}"
: "${GCS_WORK:?GCS_WORK required}"
QE_OUT="${QE_OUT:-${QE_IN%.in}.out}"
NPROC="${NPROC:-1}"
NPOOL="${NPOOL:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# -npool (k-point parallelization) speeds up dense-k pw.x runs (scf/nscf/bands) by
# splitting k-points across pools. Apply ONLY to pw.x; postproc (dos.x/projwfc.x/bands.x)
# run npool=1 (their I/O can break under pools, and they're cheap anyway).
POOLFLAG=()
if [ "$NPOOL" -gt 1 ] && [ "$(basename "$QE_BINARY")" = "pw.x" ]; then
  POOLFLAG=(-npool "$NPOOL")
fi

WORK=/scratch/work
mkdir -p "$WORK/out"
cd "$WORK" || exit 1

echo "[entrypoint] staging $QE_IN from $GCS_WORK"
gsutil -q cp "${GCS_WORK%/}/$QE_IN" "$WORK/$QE_IN"

if [ -n "${GCS_PSEUDO:-}" ]; then
  mkdir -p "$WORK/pseudo"
  echo "[entrypoint] staging pseudopotentials from $GCS_PSEUDO"
  gsutil -q -m cp -r "${GCS_PSEUDO%/}/*" "$WORK/pseudo/" || true
fi

if [ -n "${GCS_DEPS:-}" ]; then
  for dep in $GCS_DEPS; do
    echo "[entrypoint] restarting from dependency outdir $dep"
    gsutil -q -m cp -r "${dep%/}/out/*" "$WORK/out/" || true
    # pp.x artifacts (filplot/cube) live under charge/ — stage them for average.x
    gsutil -q -m cp -r "${dep%/}/charge" "$WORK/" 2>/dev/null || true
  done
fi

echo "[entrypoint] run: $QE_BINARY ${POOLFLAG[*]:-} -in $QE_IN  (np=$NPROC, omp=$OMP_NUM_THREADS, npool=$NPOOL)"
# Live monitoring: push the growing .out to GCS every 30s while QE runs, so the
# app can stream convergence points mid-run (final upload below remains canonical).
if [ "$(basename "$QE_BINARY")" = "average.x" ]; then
  # average.x reads free-form lines from stdin (not a namelist) — serial is fine.
  "$QE_BINARY" < "$QE_IN" > "$QE_OUT" 2>&1 &
elif [ "$NPROC" -gt 1 ]; then
  mpirun --allow-run-as-root -np "$NPROC" "$QE_BINARY" ${POOLFLAG[@]+"${POOLFLAG[@]}"} -in "$QE_IN" > "$QE_OUT" 2>&1 &
else
  "$QE_BINARY" ${POOLFLAG[@]+"${POOLFLAG[@]}"} -in "$QE_IN" > "$QE_OUT" 2>&1 &
fi
QE_PID=$!
(
  while kill -0 "$QE_PID" 2>/dev/null; do
    sleep 30
    [ -s "$QE_OUT" ] && gsutil -q cp "$QE_OUT" "${GCS_WORK%/}/$(basename "$QE_OUT")" 2>/dev/null || true
  done
) &
LIVE_UPLOADER=$!
wait "$QE_PID"
QE_RC=$?
kill "$LIVE_UPLOADER" 2>/dev/null || true
wait "$LIVE_UPLOADER" 2>/dev/null || true
RC=$QE_RC
echo "[entrypoint] QE exit code: $RC"

echo "[entrypoint] uploading $QE_OUT to $GCS_WORK"
gsutil -q cp "$WORK/$QE_OUT" "${GCS_WORK%/}/$QE_OUT" || true
if [ -n "$(ls -A "$WORK/out" 2>/dev/null)" ]; then
  echo "[entrypoint] uploading outdir (.save) for downstream units"
  gsutil -q -m cp -r "$WORK/out" "${GCS_WORK%/}/" || true
fi
if [ -d "$WORK/charge" ] && [ -n "$(ls -A "$WORK/charge" 2>/dev/null)" ]; then
  echo "[entrypoint] uploading charge/ (pp.x filplot + cube)"
  gsutil -q -m cp -r "$WORK/charge" "${GCS_WORK%/}/" || true
fi
shopt -s nullglob
for f in "$WORK"/*.band "$WORK"/*.band.gnu "$WORK"/*.cube "$WORK"/*.dos "$WORK"/*.pdos_* "$WORK"/*.dat; do
  gsutil -q cp "$f" "${GCS_WORK%/}/$(basename "$f")" || true
done

exit "$RC"
