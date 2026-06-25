import os
import cv2
import argparse
from pathlib import Path


def numeric_sort_key(path: Path):
    """
    按文件名中的数字排序：
    00001.png -> 1
    00010.png -> 10

    如果文件名不是纯数字，则退化为字符串排序。
    """
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    return stem


def images_to_video(
    image_dir: str,
    output_path: str,
    fps: int = 25,
    resize_to_first: bool = True,
):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_dir}")

    image_exts = {".jpg", ".jpeg", ".png"}

    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts
    ]

    image_paths = sorted(image_paths, key=numeric_sort_key)

    if len(image_paths) == 0:
        raise RuntimeError(f"文件夹内没有找到 jpg/jpeg/png 图片: {image_dir}")

    # 读取第一张图，确定视频尺寸
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise RuntimeError(f"无法读取图片: {image_paths[0]}")

    height, width = first_img.shape[:2]

    # mp4 编码
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")

    print(f"发现 {len(image_paths)} 张图片")
    print(f"输出视频: {output_path}")
    print(f"视频尺寸: {width}x{height}")
    print(f"FPS: {fps}")

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[WARN] 跳过无法读取的图片: {img_path}")
            continue

        h, w = img.shape[:2]

        if (w, h) != (width, height):
            if resize_to_first:
                img = cv2.resize(img, (width, height))
            else:
                raise RuntimeError(
                    f"图片尺寸不一致: {img_path}, "
                    f"当前尺寸 {w}x{h}, 期望尺寸 {width}x{height}"
                )

        writer.write(img)

        if (idx + 1) % 100 == 0:
            print(f"已写入 {idx + 1}/{len(image_paths)} 张")

    writer.release()
    print("视频生成完成")


def main():
    parser = argparse.ArgumentParser(
        description="将文件夹内的 jpg/png 图片按顺序生成视频"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="图片文件夹路径",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="输出视频路径，例如 output.mp4",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="视频帧率",
    )

    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="如果图片尺寸不一致，不自动 resize，而是直接报错",
    )

    args = parser.parse_args()

    images_to_video(
        image_dir=args.image_dir,
        output_path=args.output,
        fps=args.fps,
        resize_to_first=not args.no_resize,
    )


if __name__ == "__main__":
    main()