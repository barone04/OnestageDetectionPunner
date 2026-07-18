import argparse
import torch
import torch.nn as nn
import numpy as np
from models_svp import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic pruning rate search via GAM"
    )
    parser.add_argument(
        "-n",
        "--num-classes",
        type=int,
        default=10,
        choices=(10, 100),
        help="CIFAR10/100. Default: 10",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vgg_16_bn",
        choices=(
            "vgg_16_bn",
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
            "googlenet",
            "densenet40",
        ),
        help="Model. Default: vgg_16_bn",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/vgg_16_bn.pt",
        help="pretrained model path",
    )
    parser.add_argument(
        "-tr",
        "--target_rate",
        type=float,
        default=0.5,
        help="desired global compress rate, uniform for all layers. Default: 0.5",
    )

    return parser.parse_args()


def compute_singular_values(tensor):
    """
    Reshape `tensor` to 2 dimensions and return its singular values.
    """
    reshaped_tensor = tensor.view(tensor.size(0), -1)
    _, singular_values, _ = torch.linalg.svd(reshaped_tensor)

    return singular_values.numpy()


def get_prunable_weights_vgg(model):
    prunable_weights = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prunable_weights.append(module.weight.data)

    return prunable_weights


def main():
    args = parse_args()

    # Load original model
    model_ori = eval(args.model)(
        num_classes=args.num_classes, compress_rate=[0.0] * 100
    )
    ckpt = torch.load(args.ckpt, map_location=torch.device("cpu"))
    model_ori.load_state_dict(ckpt["state_dict"])

    # Get all prunable layers and its weights
    if args.model == "vgg_16_bn":
        weights = get_prunable_weights_vgg(model_ori)

    total_number_of_channels = 0
    singular_values = []
    original_number_of_channels = []
    for weight in weights:
        original_number_of_channels.append(weight.size(0))
        singular_values.append(compute_singular_values(weight))
    total_number_of_channels = np.sum(original_number_of_channels)
    total_channels_to_keep = int((1 - args.target_rate) * total_number_of_channels)
    # print(total_number_of_channels, total_channels_to_keep)

    optimal_channels = find_optimal_channels(
        singular_values, total_channels_to_keep, original_number_of_channels
    )
    print("Optimal Number of Channels to Keep:", optimal_channels)
    compress_rate = [
        (1 - x / y) for x, y in zip(optimal_channels, original_number_of_channels)
    ]
    print("Compress Rate: ", compress_rate)


def find_optimal_channels(
    singular_values, total_channels_to_keep, original_number_of_channels
):
    num_layers = len(singular_values)
    number_of_channels_to_keep = [0] * num_layers
    current_singular_values_candidates = np.array(
        [singular_value[0] for singular_value in singular_values]
    )
    for _ in range(total_channels_to_keep):
        max_idx = np.argmax(current_singular_values_candidates)
        number_of_channels_to_keep[max_idx] += 1
        if number_of_channels_to_keep[max_idx] < original_number_of_channels[max_idx]:
            current_singular_values_candidates[max_idx] = singular_values[max_idx][
                number_of_channels_to_keep[max_idx]
            ]
        else:
            current_singular_values_candidates[max_idx] = 0

    return number_of_channels_to_keep


if __name__ == "__main__":
    main()
