#!/usr/bin/env bash
# Build fp4_gemv with AddressSanitizer and ThreadSanitizer and run a small case (Linux / WSL).
set -euo pipefail
cd "$(dirname "$0")"
for s in address thread; do
  g++ -std=c++20 -O1 -g -march=native -fsanitize=$s fp4_gemv.cpp -o /tmp/fp4_gemv_$s
  echo "== -fsanitize=$s"
  /tmp/fp4_gemv_$s --experts 2 --reps 1 --threads 1,4
  rm -f /tmp/fp4_gemv_$s
done
