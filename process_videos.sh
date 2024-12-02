#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <videos_directory>"
    exit 1
fi

videos_dir="$1"
output_base_dir="${videos_dir}/colmap"

# 创建输出基础目录
mkdir -p "$output_base_dir"

# 支持的视频格式
video_formats=("*.mp4" "*.MP4" "*.avi" "*.mov" "*.MOV")

# 处理每种格式的视频
for format in "${video_formats[@]}"; do
    # 查找指定格式的视频文件
    find "$videos_dir" -maxdepth 1 -name "$format" | while read video_file; do
        if [ -f "$video_file" ]; then
            # 获取视频文件名（不含扩展名）
            video_name=$(basename "$video_file" | sed 's/\.[^.]*$//')
            output_dir="${output_base_dir}/${video_name}"
            
            # 检查对应的输出目录是否已存在
            if [ -d "$output_dir" ]; then
                echo "Skipping video: $video_file (already processed)"
                continue
            fi
            
            echo "Processing video: $video_file"
            # 运行colmap.sh处理视频
            ./colmap.sh "$video_file" "$output_base_dir"
        fi
    done
done

echo "All videos have been processed." 