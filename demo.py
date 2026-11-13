# -*- coding: utf-8 -*-
"""
批量图像分割 demo（YOLOv8-seg）

用法:
    python demo.py                         # 读 ./input，结果写到 ./output
    python demo.py --source 我的照片        # 指定输入文件夹
    python demo.py --conf 0.4              # 提高置信度阈值，只保留更确定的目标
    python demo.py --model yolov8s-seg.pt  # 换更大的模型（更准、更慢）

产出:
    output/vis/      每张图带掩码的可视化结果
    output/objects/  每个目标单独裁出来（背景涂黑）
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_bool_mask(mask, h, w):
    """把模型的掩码转成和原图同尺寸的 True/False 数组。"""
    if mask.shape != (h, w):
        resized = Image.fromarray(mask.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR)
        mask = np.array(resized)
    return mask > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="input", help="输入图片文件夹")
    parser.add_argument("--out", default="output", help="输出文件夹")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="模型权重文件")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    vis_dir = out / "vis"
    obj_dir = out / "objects"
    for d in (out, vis_dir, obj_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[!] 找不到文件夹 {src.resolve()}")
        print(f"    请先建好它，再往里放几张 jpg/png 图片：{src.resolve()}")
        return

    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"[!] {src.resolve()} 里没有图片（支持 {' '.join(sorted(IMG_EXTS))}）")
        return

    print(f"找到 {len(images)} 张图片，加载模型 {args.model} ...")
    model = YOLO(args.model)  # 第一次运行会自动下载权重

    total = 0
    for i, img_path in enumerate(images, 1):
        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        r = results[0]

        # 1) 带掩码的可视化图（r.plot() 返回 BGR，转成 RGB 再存）
        vis = r.plot()[:, :, ::-1]
        Image.fromarray(vis).save(vis_dir / f"{img_path.stem}_vis.png")

        # 2) 把每个实例单独裁出来
        n = 0
        if r.masks is not None and len(r.masks.data) > 0:
            orig = np.array(Image.open(img_path).convert("RGB"))
            h, w = orig.shape[:2]
            masks = r.masks.data.cpu().numpy()
            if r.boxes is not None and len(r.boxes) == len(masks):
                classes = r.boxes.cls.cpu().numpy().astype(int)
            else:
                classes = np.full(len(masks), -1, dtype=int)

            for mask, cls_id in zip(masks, classes):
                m = to_bool_mask(mask, h, w)
                if not m.any():
                    continue
                ys, xs = np.where(m)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                crop = orig[y0:y1, x0:x1].copy()
                crop[~m[y0:y1, x0:x1]] = 0  # 背景涂黑
                #crop[~m[y0:y1, x0:x1]] = 255   # 背景改成白色
                name = r.names.get(int(cls_id), str(cls_id))
                Image.fromarray(crop).save(obj_dir / f"{img_path.stem}_obj{n}_{name}.png")
                n += 1

        total += n
        print(f"[{i}/{len(images)}] {img_path.name}: 检出 {n} 个目标")

    print(f"\n完成：{len(images)} 张图，共 {total} 个目标")
    print(f"可视化结果: {vis_dir.resolve()}")
    print(f"裁剪结果:   {obj_dir.resolve()}")


if __name__ == "__main__":
    main()
