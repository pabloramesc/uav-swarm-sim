#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
ns3_root="$script_dir/ns-3"
source_dir="$script_dir/swarm-net-sim"
scratch_dir="$ns3_root/scratch/swarm-net-sim"

git -C "$project_root" submodule update --init ns3/ns-3

mkdir -p "$scratch_dir"
cp -R "$source_dir/." "$scratch_dir/"

cd "$ns3_root"
./ns3 configure --disable-examples --disable-tests
./ns3 build main
