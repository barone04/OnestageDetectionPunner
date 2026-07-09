import argparse
import torch
from models import *
from detection.model import (
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
)
from utils import get_cpr
from ptflops import get_model_complexity_info
from thop import profile


def parse_args():
    parser = argparse.ArgumentParser("Compute model complexity")

    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        default=10,
        choices=(10, 100, 1000, 91, 2),
        help="CIFAR10/100/Imagenet/COCO. Faster/MaskRCNN: 91. KeypointRCNN: 2. Default: 10",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet56",
        choices=(
            "vgg_16_bn",
            "vgg_19_bn",
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
            "googlenet",
            "densenet40",
            "resnet34",
            "resnet50",
            "mobilenetv2",
            "fasterrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn",
            "keypointrcnn_resnet50_fpn",
        ),
        help="Model. Default: resnet56",
    )
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]*100",
        help="list of compress rate of each layer. Default: [0.]*100",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with torch.cuda.device(0):
        compress_rate = get_cpr(args.compress_rate)
        model = eval(args.model)(
            num_classes=args.num_classes, compress_rate=compress_rate
        )

        inp_img_size = 32
        if args.num_classes == 1000:
            inp_img_size = 224
        elif args.num_classes in [91, 2]:
            inp_img_size = 800

        macs_ptfl, params_ptfl = get_model_complexity_info(
            model,
            (3, inp_img_size, inp_img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        input = torch.randn(1, 3, inp_img_size, inp_img_size)
        macs_thop, params_thop = profile(model, inputs=(input,))

        print(macs_ptfl, macs_thop)
        macs = min(macs_ptfl, macs_thop)
        params = min(params_ptfl, params_thop)

        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))

        ori_model = eval(args.model)(
            num_classes=args.num_classes, compress_rate=[0.0] * 100
        )
        ori_macs, ori_params = get_model_complexity_info(
            ori_model,
            (3, inp_img_size, inp_img_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        mac_reduced = (1 - macs / ori_macs) * 100
        param_reduced = (1 - params / ori_params) * 100
        print(f"FLOPs_reduced = {mac_reduced:.2f}")
        print(f"param_reduced = {param_reduced:.2f}")
