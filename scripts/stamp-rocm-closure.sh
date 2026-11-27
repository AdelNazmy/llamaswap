#!/bin/sh
# stamp-rocm-closure.sh — materialize the HOST ROCm runtime closure into
# build/rocm-closure/, which docker-compose.yml points the 'rocm-closure'
# build context at and the Dockerfile stamps into /opt/rocm/lib.
#
# Why the host and not the pinned rocm/dev stage: this deployment's ROCm
# (7.2.3, rocm-smi) is a CUSTOM build — real files under
# /opt/rocm/core-10.2/lib, /opt/rocm/lib entries are soname symlinks — whose
# libhipblaslt.so.1 is the TensileLite/origami generation (NEEDEDs
# libtensilelite-host.so.1 + liborigami.so.1 + librocm_kpack.so.0, loads
# compressed .dat.zlib kernels from hipblaslt/library/gfx1201/). The stock
# rocm/dev-ubuntu-24.04:7.2.3-complete libs share the version number but are
# a different build (plain TensileLibrary_lazy_gfx1201.dat layout) — mixing
# stage libs with host kernel data aborts llama-server at the first GEMM
# ("rocblaslt error: Cannot read TensileLibrary_lazy_gfx1201.dat" ->
# CUBLAS_STATUS_INTERNAL_ERROR -> exit -6). Every stamped lib must come from
# the same build as the host binaries, i.e. the host itself.
#
# Closure: walked with ldd from the GPU runtime entry points
# (libhipblas, libhipblaslt, libamdhip64, libhsa-runtime64, librocblas,
# librocsolver, librocprofiler-register, libamd_comgr, librocroller,
# libroctx64). 13 flat libs + the vendored RPATH dirs
# ($ORIGIN-relative: rocm_sysdeps/, llvm/lib, host-math/lib). Re-run on ROCm
# upgrades — same caveat as the host-built backend binaries.
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SRC=/opt/rocm/lib
DEST="$ROOT/build/rocm-closure"

if [ ! -d "$SRC" ]; then
    echo "stamp-rocm-closure: $SRC not found (non-ROCm host); staging nothing"
    mkdir -p "$DEST"
    exit 0
fi

rm -rf "$DEST"
mkdir -p "$DEST/llvm/lib" "$DEST/rocm_sysdeps/lib" "$DEST/host-math/lib"

# Flat runtime libs — cp -L dereferences the soname symlinks into /opt/rocm/lib.
for so in libamd_comgr.so.3 libamdhip64.so.7 libhipblas.so.3 libhipblaslt.so.1 \
          libhsa-runtime64.so.1 liborigami.so.1 librocblas.so.5 librocm_kpack.so.0 \
          librocprofiler-register.so.0 librocroller.so.1 librocsolver.so.0 \
          libroctx64.so.4 libtensilelite-host.so.1; do
    [ -e "$SRC/$so" ] || { echo "stamp-rocm-closure: missing $SRC/$so" >&2; exit 1; }
    cp -L "$SRC/$so" "$DEST/$so"
done

# Vendored sysdeps the libs' RPATHs resolve as $ORIGIN/rocm_sysdeps/lib
# (zlib/zstd/elf/drm/numa/lzma/bz2 — the compressed kernel support lives here).
cp -a "$SRC/rocm_sysdeps/lib/." "$DEST/rocm_sysdeps/lib/"

# llvm/lib is 2.8 GB but only libomp.so is in the closure (host OpenMP
# runtime; also the flat libomp.so radiance's plugins load via ldconfig).
cp -L "$SRC/llvm/lib/libomp.so" "$DEST/llvm/lib/libomp.so"

# host-math/lib (SuiteSparse) — referenced by $ORIGIN/host-math/lib RPATH,
# loaded on some solver paths despite not being a load-time NEEDED.
cp -a "$SRC/host-math/lib/." "$DEST/host-math/lib/"

echo "stamp-rocm-closure: staged $(du -sh "$DEST" | cut -f1) -> $DEST"
