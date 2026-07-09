import os
import torch
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from models import resnet50
from utils import get_cpr
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
    GradCAMElementWise,
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    preprocess_image,
)


def get_args():
    parser = argparse.ArgumentParser("CAM visualization")
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]*20",
        help="list of compress rate of each layer",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="checkpoint/resnet_50.pt",
        help="checkpoint path",
    )
    parser.add_argument(
        "--aug-smooth",
        action="store_true",
        help="Apply test time augmentation to smooth the CAM",
    )
    parser.add_argument(
        "--eigen-smooth",
        action="store_true",
        help="Reduce noise by taking the first principle component"
        "of cam_weights*activations",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=[
            "gradcam",
            "hirescam",
            "gradcam++",
            "scorecam",
            "xgradcam",
            "ablationcam",
            "eigencam",
            "eigengradcam",
            "layercam",
            "fullgrad",
            "gradcamelementwise",
        ],
        help="CAM method",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="input",
        help="Input directory to load the images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="CAM",
        help="Output directory to save the images",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(f"args = {args}")
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
    }
    cam_algorithm = methods[args.method]

    compress_rate = get_cpr(args.compress_rate)
    model = resnet50(compress_rate=compress_rate)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict["model"])

    class_name = args.input.split("/")[-1]
    output_dir = os.path.join(args.output, class_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if os.path.exists(args.input):
        imgs = os.listdir(args.input)
        for img in tqdm(imgs, desc="Processing images"):
            img_path = os.path.join(args.input, img)
            rgb_img, input_tensor = preprocess(img_path)
            cam_image = CAM(
                input_tensor=input_tensor,
                rgb_img=rgb_img,
                model=model,
                cam_algorithm=cam_algorithm,
                aug_smooth=args.aug_smooth,
                eigen_smooth=args.eigen_smooth,
            )

            img_name = img.split(".")[0]
            output_path = os.path.join(output_dir, f"{img_name}_{args.compress_rate}.jpg")
            cv2.imwrite(output_path, cam_image)


def preprocess(img_path):
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return rgb_img, input_tensor


def CAM(input_tensor, rgb_img, model, cam_algorithm, aug_smooth, eigen_smooth):
    target_layers = [model.layer4]
    targets = None
    with cam_algorithm(model=model, target_layers=target_layers) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image


if __name__ == "__main__":
    main()
