#!/usr/bin/env bash
# Build iobench with ThreadSanitizer and AddressSanitizer and run it on a scratch file (Linux / WSL).
set -euo pipefail
cd "$(dirname "$0")"
data=$(mktemp /tmp/iobench.XXXXXX)
trap 'rm -f "$data" /tmp/iobench_thread /tmp/iobench_address' EXIT
head -c 268435456 /dev/urandom > "$data"
for s in thread address; do
  g++ -std=c++20 -O1 -g -fsanitize=$s iobench.cpp -o /tmp/iobench_$s
  echo "== -fsanitize=$s"
  /tmp/iobench_$s "$data" --threads 1,8 --seconds 2 --block-kib 1024
done
