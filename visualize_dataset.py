import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np
import typing

colors = ["blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue", "plum"]


def plot_bbox(bbox_XYXY, label):
    xmin, ymin, xmax, ymax = bbox_XYXY
    plt.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [ymin, ymax, ymax, ymin, ymin],
        color=colors[label % len(colors)],
        label=str(label))

def read_yolo_labels(label_path: pathlib.Path, img_w: int, img_h: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Read YOLO-format label file: class x_center y_center width height (normalized floats)
    Return labels (ints) and bboxes in XYXY pixel coords (ints)
    """
    labels = []
    BBOXES_XYXY = []
    if not label_path.is_file():
        return np.array(labels), np.array(BBOXES_XYXY)
    with open(label_path, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            # convert to pixels
            cx = xc * img_w
            cy = yc * img_h
            bw = w * img_w
            bh = h * img_h
            xmin = int(round(cx - bw / 2.0))
            ymin = int(round(cy - bh / 2.0))
            xmax = int(round(cx + bw / 2.0))
            ymax = int(round(cy + bh / 2.0))
            # clamp
            xmin = max(0, min(img_w - 1, xmin))
            ymin = max(0, min(img_h - 1, ymin))
            xmax = max(0, min(img_w - 1, xmax))
            ymax = max(0, min(img_h - 1, ymax))
            labels.append(cls)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])
    return np.array(labels), np.array(BBOXES_XYXY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Dataset root (contains images/ and labels/)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Which split to visualize")
    parser.add_argument("--save-dir", type=pathlib.Path, default=None, help="Optional dir to save annotated images")
    args = parser.parse_args()

    base_path = pathlib.Path(args.directory)
    image_dir = base_path.joinpath("images", args.split)
    label_dir = base_path.joinpath("labels", args.split)
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")
    impaths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    for impath in impaths:
        label_path = label_dir.joinpath(f"{impath.stem}.txt")
        im = plt.imread(str(impath))
        # ensure image is HxW or HxWxC; matplotlib may load grayscale as HxW floats
        if im.ndim == 2:
            img_h, img_w = im.shape
            cmap = 'gray'
        else:
            img_h, img_w = im.shape[:2]
            cmap = None

        labels, bboxes_XYXY = read_yolo_labels(label_path, img_w, img_h)
        plt.figure(figsize=(6,6))
        if cmap:
            plt.imshow(im, cmap=cmap)
        else:
            plt.imshow(im)
        for bbox, label in zip(bboxes_XYXY, labels):
            plot_bbox(bbox, int(label))
        out_name = f"example_{args.split}_{impath.stem}.png"
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.save_dir.joinpath(out_name)
            plt.savefig(str(out_path))
            print(f"Saved {out_path}")
        else:
            plt.savefig(out_name)
            print(f"Saved {out_name}")
        plt.show()
