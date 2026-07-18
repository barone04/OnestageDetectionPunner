import argparse
import torch
from utils import get_cpr
from models_svp import *
import time


def get_args_parser():
    """
    Argument parser for the inference benchmark script.
    """
    parser = argparse.ArgumentParser(description="Inference benchmark")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnet50",
        choices=("resnet50"),
        help="Model to benchmark. Default: resnet50",
    )
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]*100",
        help="list of compress rate of each layer. Default: [0.]*100",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to use for inference. Default: cuda",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference. Default: 32",
    )
    parser.add_argument(
        "-t",
        "--num-tests",
        type=int,
        default=10,
        help="Number of test runs to compute the average latency. Default: 10",
    )

    return parser


def load_model(compress_rate, device):
    """
    Load the model with the specified compression rate, and move it to the appropriate device.
    """
    print(f"Loading model {args.model} with cpr {args.compress_rate}...")
    compress_rate = get_cpr(compress_rate)
    model = eval(args.model)(compress_rate=compress_rate).to(device)

    model.eval()  # Set model to evaluation mode
    print(f"Model {args.model} loaded and moved to {args.device}.")
    return model


def warmup_model(model, input_data, num_iterations=128):
    """
    Run warm-up iterations to get consistent GPU performance.
    """
    print(f"Running warm-up for {num_iterations} iterations...")
    for _ in range(num_iterations):
        _ = model(input_data)
    print("Warm-up completed.")


def benchmark_model(model, input_data, num_tests):
    """
    Benchmark the model's forward pass for a specified number of test runs and return the average latency.
    """
    latencies = []

    with torch.no_grad():
        for i in range(num_tests):
            # For GPU, synchronize before and after to get accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.time()  # Start timing

            _ = model(input_data)  # Forward pass

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()  # End timing

            # Calculate latency for this run
            latency = end_time - start_time
            latencies.append(latency)
            print(f"Test {i+1}/{num_tests}: Latency = {latency:.6f} seconds")

    # Calculate and return the average latency
    avg_latency = sum(latencies) / num_tests
    return avg_latency


def main(args):
    # Set the device (GPU or CPU)
    device = torch.device(args.device)

    # Load model and move to the appropriate device
    model = load_model(args.compress_rate, device)

    # Create random input tensor with specified batch size
    print(f"Generating random input tensor with batch size {args.batch_size}...")
    input_data = torch.randn(args.batch_size, 3, 224, 224).to(device)
    print("Input tensor generated.")

    # Warm-up the model for consistent performance
    warmup_model(model, input_data)

    # Measure and report the average latency over t runs
    print(f"Running {args.num_tests} tests for benchmarking...")
    avg_latency = benchmark_model(model, input_data, args.num_tests)
    latency_per_image = avg_latency / args.batch_size

    print(
        f"Average Latency for batch (over {args.num_tests} runs): {avg_latency:.6f} seconds"
    )
    print(f"Average Latency per image: {latency_per_image:.6f} seconds")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
