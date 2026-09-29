#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
version=${1:?usage: package-release.sh VERSION}
[[ "$version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid release version' >&2; exit 1; }
name="pearl-miner-${version}-linux-x64-cuda13"
mkdir -p dist
install -m755 target/release/pearl-miner dist/pearl-miner
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/systemd"
install -m755 target/release/pearl-miner "$stage/$name/pearl-miner"
cp README.md config.example.toml tunel.py requirements-tunnel.txt "$stage/$name/"
cp systemd/pearl-miner.service "$stage/$name/systemd/"
tar -C "$stage" -czf "dist/$name.tar.gz" "$name"
(cd dist && sha256sum pearl-miner "$name.tar.gz" > "$name.sha256")
