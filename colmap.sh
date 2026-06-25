#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <video_path> <output_path> <FPS>"
    exit 1
fi

video_path="$1"
DATASET_PATH="$2"
FPS="$3"
# 获取视频文件名（不含扩展名）作为输出文件夹名
video_name=$(basename "$video_path" | sed 's/\.[^.]*$//')
DATASET_PATH="${DATASET_PATH}/${video_name}"

mkdir -p $DATASET_PATH
mkdir -p $DATASET_PATH/images
mkdir -p $DATASET_PATH/sparse || exit 1

# ffmpeg -i $video_path -vf fps=$FPS -q:v 2 $DATASET_PATH/images/%05d.jpg || exit 1
ffmpeg -i $video_path -vf "fps=$FPS,scale=iw/2:ih/2:flags=lanczos" -q:v 2 $DATASET_PATH/images/%05d.jpg || exit 1

colmap feature_extractor \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --ImageReader.single_camera 1 || exit 1

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db || exit 1

glomap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse || exit 1

colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0 \
    --output_type TXT || exit 1

