#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def process_videos(videos_dir):
    """
    处理指定目录下的视频文件
    
    Args:
        videos_dir (str): 视频文件所在的目录路径
    """
    # 将输入路径转换为Path对象
    videos_path = Path(videos_dir)
    output_base_dir = videos_path / "colmap"
    
    # 创建输出基础目录
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_formats = [".mp4", ".MP4", ".avi", ".mov", ".MOV"]
    
    # 遍历目录下的所有文件
    for video_file in videos_path.iterdir():
        # 检查文件扩展名是否在支持的格式列表中
        if video_file.suffix in video_formats:
            # 获取视频文件名（不含扩展名）
            video_name = video_file.stem
            output_dir = output_base_dir / video_name
            
            # 检查对应的输出目录是否已存在
            if output_dir.exists():
                print(f"Skipping video: {video_file} (already processed)")
                continue
            
            print(f"Processing video: {video_file}")
            # 运行colmap.sh处理视频
            try:
                subprocess.run(["./colmap.sh", str(video_file), str(output_base_dir)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing video {video_file}: {e}")
                continue
            except FileNotFoundError:
                print("Error: colmap.sh script not found in current directory")
                sys.exit(1)
    
    print("All videos have been processed.")

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <videos_directory>")
        sys.exit(1)
    
    videos_dir = sys.argv[1]
    
    # 检查目录是否存在
    if not os.path.isdir(videos_dir):
        print(f"Error: Directory '{videos_dir}' does not exist")
        sys.exit(1)
    
    process_videos(videos_dir)

if __name__ == "__main__":
    main() 