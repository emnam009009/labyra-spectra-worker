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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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
  done
fi

echo "[entrypoint] run: $QE_BINARY -in $QE_IN  (np=$NPROC, omp=$OMP_NUM_THREADS)"
if [ "$NPROC" -gt 1 ]; then
  mpirun --allow-run-as-root -np "$NPROC" "$QE_BINARY" -in "$QE_IN" > "$QE_OUT" 2>&1
else
  "$QE_BINARY" -in "$QE_IN" > "$QE_OUT" 2>&1
fi
RC=$?
echo "[entrypoint] QE exit code: $RC"

echo "[entrypoint] uploading $QE_OUT to $GCS_WORK"
gsutil -q cp "$WORK/$QE_OUT" "${GCS_WORK%/}/$QE_OUT" || true
if [ -n "$(ls -A "$WORK/out" 2>/dev/null)" ]; then
  echo "[entrypoint] uploading outdir (.save) for downstream units"
  gsutil -q -m cp -r "$WORK/out" "${GCS_WORK%/}/" || true
fi
shopt -s nullglob
for f in "$WORK"/*.band "$WORK"/*.band.gnu "$WORK"/*.cube "$WORK"/*.dos "$WORK"/*.pdos_* "$WORK"/*.dat; do
  gsutil -q cp "$f" "${GCS_WORK%/}/$(basename "$f")" || true
done

exit "$RC"
