"""
draw_bboxes.py

Dibuja cajas de etiquetas en formato YOLO (class x_center y_center width height)
sobre imágenes.

Uso:
  - Archivo único:
      python draw_bboxes.py --image path/to/images/0.png --labels path/to/labels/0.txt
  - Carpeta (asume estructura images/ y labels/ con mismo stem):
      python draw_bboxes.py --images-dir path/to/images --labels-dir path/to/labels --out-dir annotated
  - Guardar sin mostrar:
      python draw_bboxes.py --image img.png --labels img.txt --save-out out.png --no-show
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys


def parse_yolo_label_file(path):
    """
    Lee un archivo YOLO y devuelve lista de (class, x_center, y_center, width, height)
    all floats except class may be int or float.
    """
    boxes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((cls, x_c, y_c, w, h))
    return boxes


def yolo_to_pixels(box, img_w, img_h):
    cls, x_c, y_c, w, h = box
    cx = x_c * img_w
    cy = y_c * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    # clamp
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return cls, x1, y1, x2, y2


def class_to_color(cls):
    # Predefined colors for 0-9 (BGR)
    palette = [
        (0, 0, 255),    # 0 red
        (0, 165, 255),  # 1 orange
        (0, 255, 255),  # 2 yellow
        (0, 255, 0),    # 3 green
        (255, 0, 0),    # 4 blue
        (255, 0, 255),  # 5 magenta
        (255, 255, 0),  # 6 cyan
        (128, 0, 128),  # 7 purple
        (0, 128, 255),  # 8 gold-ish
        (200, 200, 200) # 9 light gray
    ]
    return palette[cls % len(palette)]


def draw_boxes_on_image(img, boxes, thickness=2, show_labels=True, font_scale=0.6):
    """
    img: BGR image (numpy array)
    boxes: list of (cls, x1, y1, x2, y2) in pixel coords
    """
    out = img.copy()
    h, w = out.shape[:2]
    for cls, x1, y1, x2, y2 in boxes:
        color = class_to_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=thickness)
        if show_labels:
            label = str(cls)
            ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # background rectangle for text
            tx1 = x1
            ty1 = y1 - int(th * 1.2) - 4
            if ty1 < 0:
                ty1 = y1 + 4
            tx2 = tx1 + tw + 6
            ty2 = ty1 + th + 4
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), color, thickness=-1)
            cv2.putText(out, label, (tx1 + 3, ty2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return out


def process_one(image_path: Path, label_path: Path, save_out: Path = None, show=True, scale=1.0):
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed reading image: {image_path}", file=sys.stderr)
        return
    img_h, img_w = img.shape[:2]

    if not label_path.exists():
        print(f"Label file not found, skipping: {label_path}", file=sys.stderr)
        return

    boxes_yolo = parse_yolo_label_file(label_path)
    box_pixels = [yolo_to_pixels(b, img_w, img_h) for b in boxes_yolo]
    annotated = draw_boxes_on_image(img, box_pixels)

    if scale != 1.0:
        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))
        annotated_disp = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        annotated_disp = annotated

    if save_out:
        save_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_out), annotated)
        print(f"Saved annotated image to: {save_out}")

    if show:
        winname = f"Annotated: {image_path.name}"
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, annotated_disp)
        print("Press any key to continue (or close the window).")
        cv2.waitKey(0)
        cv2.destroyWindow(winname)


def main():
    parser = argparse.ArgumentParser(description="Draw YOLO-format bboxes on images.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image file")
    group.add_argument("--images-dir", type=Path, help="Directory with images to process (png/jpg)")
    parser.add_argument("--labels", type=Path, help="Path to label file for single image (if --image used)")
    parser.add_argument("--labels-dir", type=Path, help="Directory with label files (default: images-dir/../labels or images-dir/labels)")
    parser.add_argument("--out-dir", type=Path, help="Directory to save annotated images (optional)")
    parser.add_argument("--no-show", action="store_true", help="Do not show images interactively")
    parser.add_argument("--scale", type=float, default=0.5, help="Display scale for showing images (default 0.5)")
    parser.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg"], help="Image extensions to consider when using --images-dir")
    args = parser.parse_args()

    if args.image:
        label_path = args.labels if args.labels else args.image.with_suffix(".txt")
        save_out = args.out_dir.joinpath(args.image.name) if args.out_dir else None
        process_one(args.image, label_path, save_out=save_out, show=not args.no_show, scale=args.scale)
        return

    images_dir = args.images_dir
    if not images_dir.exists():
        print("Images directory not found.", file=sys.stderr)
        return

    # determine default labels dir if not provided
    if args.labels_dir:
        labels_dir = args.labels_dir
    else:
        # common layouts: parent/labels or images_dir/labels
        candidate = images_dir.parent.joinpath("labels")
        if candidate.exists():
            labels_dir = candidate
        else:
            labels_dir = images_dir.joinpath("labels")

    out_dir = args.out_dir

    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in [e.lower() for e in args.ext]:
            continue
        label_path = labels_dir.joinpath(p.with_suffix(".txt").name)
        save_out = out_dir.joinpath(p.name) if out_dir else None
        process_one(p, label_path, save_out=save_out, show=not args.no_show, scale=args.scale)


if __name__ == "__main__":
    main()