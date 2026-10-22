#!/bin/sh
# build_engines.sh: build colibri v2.0.0's DeepSeek V4 engine twice with the same compiler,
# once for the shipped instruction set (x86-64-v3, AVX2) and once for Zen 5 (znver5, AVX-512),
# so the instruction set is the only difference between the two.
# Then the same two builds again with layer_region.patch applied (one OpenMP parallel region
# per layer for the routed experts), in layer-region/<arch>.
# Needs llvm-mingw (clang + LLVM libomp, which it ships only as a DLL, hence no -static), gh and patch.
# Usage (Git Bash): LLVM_MINGW=/c/path/to/llvm-mingw sh build_engines.sh /c/path/to/build-dir
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?build directory}
: "${LLVM_MINGW:?path to llvm-mingw}"
export PATH="$LLVM_MINGW/bin:$PATH"
mkdir -p "$out"
cd "$out"
gh api repos/JustVugg/colibri/tarball/v2.0.0 > src.tar.gz
tar xzf src.tar.gz
src=$(ls -d JustVugg-colibri-*/c)
mkdir -p layer-region
cp -r "$src" layer-region/c
(cd layer-region && patch -p1 -s < "$here/layer_region.patch")
build() {  # build <source dir> <arch> <destination>
    cp -r "$1" "$3"
    (cd "$3" && mingw32-make -f Makefile.deepseek-v4 deepseek-v4 -j12 CC=clang ARCH="$2" \
        LDFLAGS="-lm -fopenmp -pthread -flto" > build.log 2>&1)
    cp "$LLVM_MINGW/x86_64-w64-mingw32/bin/libomp.dll" "$LLVM_MINGW/x86_64-w64-mingw32/bin/libwinpthread-1.dll" "$3/"
    echo "$3: $(llvm-objdump -d --no-show-raw-insn "$3/deepseek_v4.exe" | grep -c zmm) AVX-512 (zmm) instructions"
}
for arch in x86-64-v3 znver5; do
    build "$src" "$arch" "$arch"
    build layer-region/c "$arch" "layer-region/$arch"
done
