#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: download_autovi_assets.sh [options]

Downloads the AutoVI Zenodo archives referenced in docs/datasheet and optionally
extracts a lightweight PNG sample for quick tests.

Options:
  --samples N      Copy the first N image files per category into docs/samples/.
                   Implies extraction. Default: disabled.
  --no-extract     Skip unzip step (keep only the downloaded archives).
  --force          Re-download archives even if they already exist.
  -h, --help       Show this help and exit.

Environment variables:
  AUTO_VI_DOWNLOAD_DIR   Directory for downloaded zip files (default: data/downloads).
  AUTO_VI_EXTRACT_DIR    Directory for extracted archives (default: data/raw).
  AUTO_VI_SAMPLES_DIR    Directory for sample exports (default: docs/samples).
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${AUTO_VI_DOWNLOAD_DIR:-$ROOT_DIR/data/downloads}"
EXTRACT_DIR="${AUTO_VI_EXTRACT_DIR:-$ROOT_DIR/data/raw}"
SAMPLES_DIR="${AUTO_VI_SAMPLES_DIR:-$ROOT_DIR/docs/samples}"

SAMPLE_COUNT=0
DO_EXTRACT=true
FORCE_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples)
      [[ $# -ge 2 ]] || { echo "--samples requires an argument" >&2; exit 1; }
      SAMPLE_COUNT="$2"
      shift 2
      ;;
    --no-extract)
      DO_EXTRACT=false
      shift
      ;;
    --force)
      FORCE_DOWNLOAD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if (( SAMPLE_COUNT > 0 )); then
  DO_EXTRACT=true
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd curl
require_cmd unzip

HASH_TOOL=""
if command -v sha256sum >/dev/null 2>&1; then
  HASH_TOOL="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  HASH_TOOL="shasum -a 256"
else
  echo "Missing sha256sum or shasum" >&2
  exit 1
fi

calc_sha256() {
  if [[ "$HASH_TOOL" == "sha256sum" ]]; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

DATASETS=(
  engine_wiring
  pipe_clip
  pipe_staple
  tank_screw
  underbody_pipes
  underbody_screw
)

dataset_url() {
  case "$1" in
    engine_wiring) echo "https://zenodo.org/records/10459003/files/engine_wiring.zip" ;;
    pipe_clip) echo "https://zenodo.org/records/10459003/files/pipe_clip.zip" ;;
    pipe_staple) echo "https://zenodo.org/records/10459003/files/pipe_staple.zip" ;;
    tank_screw) echo "https://zenodo.org/records/10459003/files/tank_screw.zip" ;;
    underbody_pipes) echo "https://zenodo.org/records/10459003/files/underbody_pipes.zip" ;;
    underbody_screw) echo "https://zenodo.org/records/10459003/files/underbody_screw.zip" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

dataset_hash() {
  case "$1" in
    engine_wiring) echo "252590d3249f7fbdf83a7e9b735ef0df175adc218460247e9a63bef1c03d420c" ;;
    pipe_clip) echo "955bb17b3a471e23979f46f998a4723acd00cf3afc4d5edf0578dfb6ea80d6c3" ;;
    pipe_staple) echo "fb9287f2cc86d660310e9886fdebbe2bd17269e853e29d06d708c6a996df1b18" ;;
    tank_screw) echo "48d7193164b36de03cc10c9f7b1b64ea98a0ce7aa57867c8f1d96341c497b4b0" ;;
    underbody_pipes) echo "fc1e53336d46fb2317d71e95c011bac012b609f2898fdb02780c920c19a113c7" ;;
    underbody_screw) echo "3e9bf6a43033a22c7c9f927a43c392a3b037d22580a27a48d379d4504d9cc6cf" ;;
    *) echo "Unknown dataset: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$DOWNLOAD_DIR"
$DO_EXTRACT && mkdir -p "$EXTRACT_DIR"
(( SAMPLE_COUNT > 0 )) && mkdir -p "$SAMPLES_DIR"

for category in "${DATASETS[@]}"; do
  url="$(dataset_url "$category")"
  archive="$DOWNLOAD_DIR/${category}.zip"
  if [[ -f "$archive" && $FORCE_DOWNLOAD == false ]]; then
    echo "[skip] $archive already exists"
  else
    echo "[download] $category"
    curl -L --retry 3 --retry-all-errors -o "$archive" "$url"
  fi

  echo "[verify] $category"
  actual="$(calc_sha256 "$archive")"
  expected="$(dataset_hash "$category")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $category" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi

  if $DO_EXTRACT; then
    target_dir="$EXTRACT_DIR/$category"
    echo "[extract] $category -> $target_dir"
    mkdir -p "$target_dir"
    unzip -q -o "$archive" -d "$target_dir"

    if (( SAMPLE_COUNT > 0 )); then
      sample_dir="$SAMPLES_DIR/$category"
      mkdir -p "$sample_dir"
      echo "[sample] copying first $SAMPLE_COUNT PNG files for $category"
      sample_list=$(find "$target_dir" -type f -iname '*.png' | sort | head -n "$SAMPLE_COUNT")
      if [[ -z "$sample_list" ]]; then
        echo "No PNG files found in $target_dir; skipping samples" >&2
      else
        echo "$sample_list" | while IFS= read -r file_path; do
          [[ -z "$file_path" ]] && continue
          cp "$file_path" "$sample_dir/"
        done
      fi
    fi
  fi
done

echo "Done. Archives stored in $DOWNLOAD_DIR"
$DO_EXTRACT && echo "Extracted payloads under $EXTRACT_DIR"
(( SAMPLE_COUNT > 0 )) && echo "Samples exported to $SAMPLES_DIR"
