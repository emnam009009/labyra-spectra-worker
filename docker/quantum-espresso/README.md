# Quantum ESPRESSO image — Labyra Cloud Batch DFT

Runtime container that Google Cloud Batch pulls per DFT task (compute backend
**cloud-batch**). The default backend **generate-only** runs QE on the user's own
machine and never needs this image. Infra — separate from the worker service.

## Build & push
```bash
cd docker/quantum-espresso
REG=REGION-docker.pkg.dev/PROJECT/REPO        # e.g. asia-southeast1-docker.pkg.dev/labyra/dft
docker build -t "$REG/quantum-espresso:7.4.1" .
docker push  "$REG/quantum-espresso:7.4.1"
```
Built from official **QEF/q-e** source (tag `qe-7.4.1`), CPU + OpenMPI. The `imageUri`
you push goes into the Batch manifest (`batch_client.build_batch_job(image_uri=…)`).

### Faster dev alternative (not recommended for prod)
A prebuilt `.deb` (GNU + OpenMPI, Ubuntu 24.04) exists at
`github.com/pranabdas/espresso/releases` (`quantum-espresso_7.4-1_amd64.deb`,
sha256 `2da740…c9030`). Swap the build stage for `apt-get install ./qe.deb`. It's QE
**7.4** (not 7.4.1) from a personal repo — fine for local iteration, but source-build
is the trustworthy production path.

### GPU (deferred)
`high-gpu` preset (NVIDIA **L4** = Ada, compute capability **8.9**) needs a separate
NVHPC+CUDA image: `cmake -DQE_ENABLE_CUDA=ON … --with-cuda-cc=89`. Build after the CPU
MVP is exercised.

## Entrypoint contract (env-driven — the worker driver sets these per task)
| Env | Meaning |
|---|---|
| `QE_BINARY` | `pw.x` \| `bands.x` \| `dos.x` \| `projwfc.x` \| `pp.x` (required) |
| `QE_IN` | input filename, e.g. `pw_scf.in` (required) |
| `QE_OUT` | output filename (default `<QE_IN>.out`) |
| `GCS_WORK` | gs:// prefix for this unit (`.in` here; `.out` + `out/` written here) (required) |
| `GCS_DEPS` | space-separated gs:// prefixes whose `out/` to restart from (scf→nscf/bands/dos/pdos/charge) |
| `GCS_PSEUDO` | gs:// prefix with pseudopotential UPF files (optional) |
| `NPROC` | MPI ranks (default 1) |
| `OMP_NUM_THREADS` | OpenMP threads (default 1) |

**Flow:** localize (`.in` + pseudo + dep `out/`) → run QE on **local SSD** scratch
(`/scratch/work`) → upload `.out` + `out/` (`.save`) + artifacts (`.band/.cube/.dos/.pdos_*`)
to `GCS_WORK`. The `.out` is uploaded even on failure (diagnosis); the container exits
with QE's own code so Batch `lifecyclePolicies` (FAIL_TASK on 1/137/139) fire.

> **Generator contract:** the generated `.in` must set `outdir = './out'` (relative) so
> QE scratch maps to the staged local dir and the `.save` handoff round-trips via GCS.

## IAM
The Batch service account needs `roles/batch.jobsEditor`; the task's compute SA needs
read/write on the workflow + pseudo buckets (`roles/storage.objectAdmin` scoped to them).
