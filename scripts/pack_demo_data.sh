#!/usr/bin/env bash
# Pack demo data into a compressed tar for sharing / download.
# Usage: bash scripts/pack_demo_data.sh [output_path]
set -e

OUT="${1:-demo_data.tar.gz}"

tar czf "$OUT" \
    demo/images/ \
    demo/preprocessed/ \
    demo/vlm_estimate_headcrop/ \
    demo/multihmr_multiHMR_672_L_anny/ \
    demo/person_dataset_10_headcrop/

echo "Packed demo data to $OUT ($(du -sh "$OUT" | cut -f1))"
