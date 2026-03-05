#!/bin/bash
set -e

cd "$(cd "$(dirname "$0")" && pwd)/.."

cargo test --release --workspace
