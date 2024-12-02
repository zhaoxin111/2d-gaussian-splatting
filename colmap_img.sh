#!/bin/bash

DATASET_PATH="$1"

colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --ImageReader.single_camera 1

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir -p $DATASET_PATH/sparse

glomap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    # --Mapper.init_min_tri_angle 2 \
    # --Mapper.ba_refine_principal_point 1

colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --output_type TXT

