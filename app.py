# -*- coding: utf-8 -*-
"""
Gradio 界面：上传图片 → 自动分割 → 显示结果和统计

运行:
    python app.py
然后浏览器打开 http://127.0.0.1:7860
"""
import numpy as np
import gradio as gr
from ultralytics import YOLO

MODEL = YOLO("yolov8n-seg.pt")


def segment(image, conf):
    if image is None:
        return None, "请先上传一张图片"

    # 直接把 PIL 图交给模型，ultralytics 会自己处理颜色通道
    r = MODEL.predict(image.convert("RGB"), conf=conf, verbose=False)[0]

    vis = r.plot()[:, :, ::-1]  # BGR -> RGB

    n = 0 if r.masks is None else len(r.masks.data)
    labels = []
    if r.boxes is not None and len(r.boxes) > 0:
        for c in r.boxes.cls.cpu().numpy().astype(int):
            labels.append(r.names.get(int(c), str(c)))

    if n:
        uniq = "、".join(sorted(set(labels)))
        summary = f"检出 {n} 个目标：{uniq}"
    else:
        summary = "没有检出目标（可以试试调低置信度阈值）"

    return vis, summary


demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度阈值"),
    ],
    outputs=[
        gr.Image(label="分割结果"),
        gr.Textbox(label="统计"),
    ],
    title="图像分割 Demo · YOLOv8-seg",
    description="上传图片，模型自动分割出图中的目标并标注类别。",
)

if __name__ == "__main__":
    demo.launch()
