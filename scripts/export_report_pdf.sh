#!/bin/bash
set -euo pipefail
DD_REPORT_DIR=$(realpath "${1:?Usage: export_report_pdf.sh REPORT_DIR}")
DD_BROWSER=${DD_BROWSER:-/usr/lib64/chromium-browser/chromium-browser}
export XDG_CACHE_HOME="$DD_REPORT_DIR/pdf_browser/cache"
export XDG_CONFIG_HOME="$DD_REPORT_DIR/pdf_browser/config"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
"$DD_BROWSER" --headless --no-sandbox --disable-gpu --disable-dev-shm-usage \
 --disable-background-networking --allow-file-access-from-files \
 --run-all-compositor-stages-before-draw --virtual-time-budget=15000 \
 --user-data-dir="$DD_REPORT_DIR/pdf_browser/profile" --no-pdf-header-footer \
 --print-to-pdf="$DD_REPORT_DIR/report.pdf" "file://$DD_REPORT_DIR/index.html"
