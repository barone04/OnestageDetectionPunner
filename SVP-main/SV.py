import torch
import tensorly
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np
from itertools import product
import time
import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Change Point Guided Channel Selection")
    parser.add_argument("-n", "--layers", type=int, default=13, help="num of layers")

    return parser.parse_args()


def compute_core_norms(tensor):
    """
    Decompose `tensor` with full-rank HOSVD.
    Compute norm of each slice of the `core` tensor
    """
    core, _ = tucker(tensor, rank=tensor.shape)
    norms = torch.norm(core.reshape(tensor.size(0), -1), dim=1)

    return norms


def generate_tensor(
    C_out=512, C_in=256, k_h=3, k_w=3, pruning_rate=0.5, noise_amplitude=0.01
):
    """
    Generate a tensor with specified dimensions.

    Args:
        C_out (int): Number of output channels.
        C_in (int): Number of input channels.
        k_h (int): Height of the kernel.
        k_w (int): Width of the kernel.
        pruning_rate (float): Pruning rate, ranging from 0 to 1.
        noise_amplitude (float): Amplitude of the noise.

    Returns:
        torch.Tensor: Generated tensor.
    """
    # Calculate the number of channels to keep after pruning
    num_channels_to_keep = round(C_out * (1 - pruning_rate))

    # Initialize tensor with random values for the first part
    tensor = torch.randn(num_channels_to_keep, C_in, k_h, k_w)

    # Calculate the number of channels to copy
    num_channels_to_copy = C_out - num_channels_to_keep

    # Copy values from the first part and add noise
    for i in range(0, num_channels_to_copy, num_channels_to_keep):
        end_index = min(num_channels_to_keep, num_channels_to_copy - i)
        tensor = torch.cat(
            (
                tensor,
                tensor[:end_index]
                + noise_amplitude * torch.randn_like(tensor[:end_index]),
            )
        )

    return tensor


def reset_data(N=13):
    channels = [
        64,
        64,
        128,
        128,
        256,
        256,
        256,
        512,
        512,
        512,
        512,
        512,
        512,
    ]

    pruning_rates = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.9,
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    # Initialize C_in for the first layer
    C_in = 64

    # Initialize a list to store the generated tensors
    vgg_layers = []

    # Loop over the channels and pruning_rates
    for C_out, pruning_rate in zip(channels, pruning_rates):
        # Generate tensor for the current layer
        layer_tensor = generate_tensor(
            C_out=C_out, C_in=C_in, pruning_rate=pruning_rate
        )

        # Update C_in for the next layer
        C_in = C_out

        # Append the generated tensor to the list
        vgg_layers.append(layer_tensor)

    # Display the generated tensors
    for i, layer_tensor in enumerate(vgg_layers):
        print(f"Layer {i + 1}: {layer_tensor.shape}")

    # Number of layers
    channels = channels[:N]
    vgg_layers = vgg_layers[:N]
    pruning_rates = pruning_rates[:N]
    singular_values = []
    # Replace this with your actual singular values for each layer
    for i in vgg_layers:
        SV = compute_core_norms(i).numpy()
        singular_values.append(SV)
    # # Plotting singular values for each layer
    # for i, sv in enumerate(singular_values):
    #     layer_number = i + 1  # Adjust for 1-based indexing
    #     plt.plot(sv, label=f"Layer {layer_number}")

    # # Adding labels and legend
    # plt.xlabel("Number of filters")
    # plt.ylabel("Singular Value")
    # plt.title(f"Singular Values for N={N} Layers")
    # plt.legend()

    # Total number of original channels
    total_original_channels = np.sum(channels)

    # Calculate the absolute number of channels to remove for each layer
    channels_to_remove = [round(pr * ch) for pr, ch in zip(pruning_rates, channels)]

    # Calculate the sum of the absolute number of channels to remove
    total_channels_to_remove = np.sum(channels_to_remove)

    # Calculate the absolute number of channels to keep for each layer
    gt_channels_to_keep = [
        round((1 - pr) * ch) for pr, ch in zip(pruning_rates, channels)
    ]

    total_channels_to_keep = total_original_channels - total_channels_to_remove

    print("Total Original Channels:", total_original_channels)
    print("Total Channels to Remove:", total_channels_to_remove)
    print(f"Total Channels to Keep: {total_channels_to_keep}")

    print(f"Groundtruth channels_to_keep = {gt_channels_to_keep}")

    return (
        singular_values,
        channels,
        vgg_layers,
        total_original_channels,
        channels_to_remove,
        total_channels_to_keep,
        gt_channels_to_keep,
    )


def calculate_objective_value(channels_to_keep, change_points_info):
    return np.sum([change_points_info[i][ch] for i, ch in enumerate(channels_to_keep)])


def topk_change_points(signal, k=4):
    """
    Apply change points detection for signal.
    Return top k change points.
    """
    algo = rpt.Pelt(model="l1", jump=1).fit(signal)
    change_points = algo.predict(pen=10)
    change_points = np.array(change_points)
    differences = -np.diff(signal[change_points - 1])
    topk = min(k, len(differences))
    topk_diff_indices = differences.argsort()[-topk:][::-1]

    return change_points[topk_diff_indices]


def detect_change_points(singular_values):
    """
    Apply change points detection for singular values of each layer.
    Return change points and corresponding sum of singular values for each layer.
    """
    change_points = []
    change_points_info = []

    for layer_num, singular_value in enumerate(singular_values, 1):
        algo = rpt.Pelt(model="l1", jump=1).fit(singular_value)
        points = topk_change_points(singular_value)

        # Compute the sum of singular values at change points
        info = {point: np.sum(singular_value[:point]) for point in points}
        print(
            f"Layer {layer_num}: Detected {len(points)} change points "
            f"at positions {points}, "
            f"with corresponding sums {info}"
        )
        change_points.append(points)
        change_points_info.append(info)

    return change_points, change_points_info


def find_optimal_channels(singular_values, total_channels_to_keep):
    start_time = time.time()

    best_channels_to_keep = None
    best_objective_value = float("-inf")

    # Detect change points
    change_points, change_points_info = detect_change_points(singular_values)
    print(change_points_info[0])

    # Generate combinations iteratively
    combinations = product(*change_points)

    # Find the combination with the maximum objective value
    for channels_to_keep in combinations:
        # Check if the total number of channels to keep is equal to the target
        if sum(channels_to_keep) <= total_channels_to_keep:
            current_objective_value = calculate_objective_value(
                channels_to_keep, change_points_info
            )
            if current_objective_value > best_objective_value:
                best_objective_value = current_objective_value
                best_channels_to_keep = channels_to_keep

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time} seconds")

    return best_channels_to_keep, execution_time


if __name__ == "__main__":
    tensorly.set_backend("pytorch")
    args = parse_args()
    wandb.init(
        name=f"N={args.layers}",
        project=f"Change Point Guided Channel Selection",
        config=vars(args),
    )
    (
        singular_values,
        channels,
        vgg_layers,
        total_original_channels,
        channels_to_remove,
        total_channels_to_keep,
        gt_channels_to_keep,
    ) = reset_data(N=args.layers)
    # Assuming singular_values, total_channels_to_keep, and channels are defined
    optimal_channels, execution_time = find_optimal_channels(
        singular_values, total_channels_to_keep
    )
    print("Optimal Channels to Keep:", optimal_channels)
    assert (
        optimal_channels == np.array(gt_channels_to_keep)
    ).all(), (
        f"Prediction {optimal_channels} differs from groundtruth {gt_channels_to_keep}"
    )

    wandb.log(
        {
            "execution_time": execution_time,
        }
    )
