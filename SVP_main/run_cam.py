import subprocess
import argparse

def run_cam_commands(input):
    commands = [
        f"python cam.py --checkpoint imagenet_resnet50_.0.20_76.15.pt -cpr [0.]*20 -i ~/datasets/val/{input}",
        f"python cam.py --checkpoint imagenet_resnet50_.0.255.20_77.34.pt -cpr [0.255]*20 -i ~/datasets/val/{input}",
        f"python cam.py --checkpoint imagenet_resnet50_.0.4.20_75.96.pt -cpr [0.4]*20 -i ~/datasets/val/{input}",
        f"python cam.py --checkpoint imagenet_resnet50_.0.55.20_73.88.pt -cpr [0.55]*20 -i ~/datasets/val/{input}"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAM commands with different input folders")
    parser.add_argument("input", help="Name of the input folder")
    args = parser.parse_args()

    run_cam_commands(args.input)
