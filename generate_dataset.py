import argparse
import mnist
import pathlib
import cv2
import numpy as np
import tqdm


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(
            calculate_iou(bbox, other_bbox)
        )
    return ious


def tight_bbox(digit, orig_bbox):
    xmin, ymin, xmax, ymax = orig_bbox
    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift
    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift
    ymin
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    # dirpath is expected to be base/<split>; real image dir will be base/images/<split>
    parent = dirpath.parent
    split = dirpath.name
    image_dir = parent.joinpath("images", split)
    label_dir = parent.joinpath("labels", split)
    if not image_dir.is_dir() or not label_dir.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {parent}"
        impath = image_dir.joinpath(f"{image_id}.png")
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
        label_path = label_dir.joinpath(f"{image_id}.txt")
        assert label_path.is_file(),  f"{error_msg} {impath} \n\t{error_msg2}"
    return True


def generate_dataset(dirpath: pathlib.Path,
                     num_images: int,
                     max_digit_size: int,
                     min_digit_size: int,
                     imsize: int,
                     max_digits_per_image: int,
                     mnist_images: np.ndarray,
                     mnist_labels: np.ndarray,
                     regenerate_dataset: bool):
    if dataset_exists(dirpath, num_images) and not regenerate_dataset:
        return
    max_image_value = 255
    assert mnist_images.dtype == np.uint8
    # dirpath is base/<split>; create images/<split> and labels/<split>
    parent = dirpath.parent
    split = dirpath.name
    image_dir = parent.joinpath("images", split)
    label_dir = parent.joinpath("labels", split)
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving"):
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []
        num_images = np.random.randint(0, max_digits_per_image)
        for _ in range(num_images+1):
            while True:
                width = np.random.randint(min_digit_size, max_digit_size)
                x0 = np.random.randint(0, imsize-width)
                y0 = np.random.randint(0, imsize-width)
                ious = compute_iou_all([x0, y0, x0+width, y0+width], bboxes)
                if max(ious) < 0.25:
                    break
            digit_idx = np.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(np.float32)
            digit = cv2.resize(digit, (width, width))
            label = mnist_labels[digit_idx]
            labels.append(label)
            assert im[y0:y0+width, x0:x0+width].shape == digit.shape, \
                f"imshape: {im[y0:y0+width, x0:x0+width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [x0, y0, x0+width, y0+width])
            bboxes.append(bbox)

            im[y0:y0+width, x0:x0+width] += digit
            im[im > max_image_value] = max_image_value
        image_target_path = image_dir.joinpath(f"{image_id}.png")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")
        im = im.astype(np.uint8)
        cv2.imwrite(str(image_target_path), im)
        # Write labels in YOLO format: "class x_center y_center width height"
        # all coordinates normalized to [0,1] relative to image size (imsize)
        # class equals the digit label
        with open(label_target_path, "w") as fp:
            for l, bbox in zip(labels, bboxes):
                xmin, ymin, xmax, ymax = bbox
                x_center = (xmin + xmax) / 2.0 / imsize
                y_center = (ymin + ymax) / 2.0 / imsize
                width = (xmax - xmin) / imsize
                height = (ymax - ymin) / imsize
                # YOLO expects: class x_center y_center width height (space-separated, floats)
                fp.write(f"{l} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


if __name__ == "__main__":
    dataset_name = "mnist_object_detection"
    base_path = f"datasets/{dataset_name}"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imsize", default=300, type=int
    )
    parser.add_argument(
        "--max-digit-size", default=100, type=int
    )
    parser.add_argument(
        "--min-digit-size", default=15, type=int
    )
    parser.add_argument(
        "--num-train-images", default=10000, type=int
    )
    parser.add_argument(
        "--num-test-images", default=1000, type=int,
        help="Number of test images to generate from the MNIST test split"
    )
    parser.add_argument(
        "--val-split", default=0.1, type=float,
        help="Fraction of the generated train set to use as validation (0..1)"
    )
    parser.add_argument(
        "--max-digits-per-image", default=20, type=int
    )
    parser.add_argument(
        "--regenerate-datasets", default=False, type=bool
    )
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = mnist.load()
    # Generate train+val (from MNIST train split) and a separate test (from MNIST test split)
    base = pathlib.Path(base_path)

    # compute train/val split counts
    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in range [0, 1)")

    total = args.num_train_images
    val_count = int(round(total * args.val_split))
    train_count = total - val_count

    if train_count <= 0:
        raise ValueError("val-split too large, no training images would remain")

    print(f"Generating {train_count} train images and {val_count} val images (val_split={args.val_split})")

    generate_dataset(
        base.joinpath("train"),
        train_count,
        args.max_digit_size,
        args.min_digit_size,
        args.imsize,
        args.max_digits_per_image,
        X_train,
        Y_train,
        args.regenerate_datasets
    )

    generate_dataset(
        base.joinpath("val"),
        val_count,
        args.max_digit_size,
        args.min_digit_size,
        args.imsize,
        args.max_digits_per_image,
        X_train,
        Y_train,
        args.regenerate_datasets
    )

    # Test dataset from MNIST test split (may be smaller than requested if MNIST test size limit reached)
    num_test = args.num_test_images
    max_available = len(X_test)
    if num_test > max_available:
        print(f"Requested {num_test} test images but only {max_available} are available in MNIST test split. Reducing to {max_available}.")
        num_test = max_available

    generate_dataset(
        base.joinpath("test"),
        num_test,
        args.max_digit_size,
        args.min_digit_size,
        args.imsize,
        args.max_digits_per_image,
        X_test,
        Y_test,
        args.regenerate_datasets
    )

    # Create a dataset YAML index (YOLO-style) at the dataset root
    dataset_root = base
    yaml_path = dataset_root.joinpath("data.yaml")
    names = {i: str(i) for i in range(10)}
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {dataset_name}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n")
        yf.write("test: images/test\n")
        yf.write("\n")
        yf.write("names:\n")
        for k, v in names.items():
            yf.write(f"    {k}: '{v}'\n")
    print(f"Wrote dataset YAML to: {yaml_path}")
