#!/usr/bin/env bash
source "$(dirname "$0")/../setup.sh"

python -m preprocess.build_test_dataset \
    --data_root demo \
    --dataset_name multi_person \
    --preprocess_data
