#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p bin
for tool in look_pipeline stack_burst bird_focus; do
  swiftc -O "Sources/$tool.swift" -o "bin/$tool"
done
echo "built: bin/look_pipeline bin/stack_burst bin/bird_focus"
