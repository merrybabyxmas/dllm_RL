#!/bin/bash
# Quick smoke test: run pytest unit tests only (no GPU required)
set -e

PACKAGE_DIR="/home/dongwoo43/papers/paper_dllm/confidence_credit_dllm_rl"

cd "${PACKAGE_DIR}"

echo "Running cc_rl smoke tests..."
python -m pytest tests/ -v --tb=short

echo "All smoke tests passed."
