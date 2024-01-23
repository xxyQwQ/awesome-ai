import cv2
import numpy as np
import os
from PIL import Image
from .utils import logging

def make_video(folder):
    image_sequence = []

    while os.path.exists(f"{folder}/{len(image_sequence)}.png"):
        image_sequence.append(f"{folder}/{len(image_sequence)}.png")

    fade_duration = 0.05
    fps = 30
    fade_frames = int(fade_duration * fps + 1)  # 渐变持续的帧数

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(folder, "animation.mp4")
    out = None

    last_image = Image.open(image_sequence[-1])
    canvas_size = last_image.size
    tmp_sequence = []
    try:
        logging.info("Making video...")
        for i, img_path in enumerate(image_sequence):
            img = Image.open(img_path).convert("RGBA")
            new_canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 1))
            new_canvas.paste(img, (0, 0))
            temp_path = f"{folder}/temp_{i}.png"
            new_canvas.save(temp_path)
            tmp_sequence.append(temp_path)
            
        for i in range(len(tmp_sequence) - 1):
            img1 = cv2.imread(tmp_sequence[i])
            img2 = cv2.imread(tmp_sequence[i + 1])

            if out is None:
                h, w, _ = img1.shape
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for f in range(fade_frames + 1):
                alpha = f / fade_frames
                frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
                out.write(frame)

        out.release()
    except Exception as e:
        logging.error(e)
        output_path = None
    for i in range(len(tmp_sequence)):
        os.remove(f"{folder}/{i}.png")
        os.remove(f"{folder}/temp_{i}.png")
    return output_path