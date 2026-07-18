import os
import argparse
import time
from accelerate import Accelerator
import torch
import torch.nn as nn
from models_svp import *
from utils import get_cpr
from data import get_dataloaders
import wandb
import ast
import copy
from train import test


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10/100 Sweeping")
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
        ),
        help="Model. Default: resnet56",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=600,
        type=int,
        metavar="N",
        help="number of total epochs to run.Default: 600",
    )
    parser.add_argument("--dataset-path", type=str, default=os.getenv("DATASETS"))
    parser.add_argument(
        "--bs",
        type=ast.literal_eval,
        default=[128, 256],
        help="list of batch sizes",
    )
    parser.add_argument(
        "--lr",
        type=ast.literal_eval,
        default=[0.5, 0.1, 0.05, 0.01, 0.005],
        help="list of learning rates",
    )
    parser.add_argument(
        "--wd",
        type=ast.literal_eval,
        default=[2e-5, 5e-5],
        help="list of weight decays",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-min", type=float, default=0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--eras", type=int, default=1)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "-cpr",
        "--compress_rate",
        type=str,
        default="[0.]*100",
        help="list of compress rate of each layer. Default: [0.]*100",
    )
    parser.add_argument(
        "-t",
        "--target",
        default=99.99,
        type=float,
        help="target accuracy. Default: 99.99",
    )

    return parser.parse_args()


args = parse_args()
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "top1"},
    "parameters": {
        "bs": {"values": args.bs},
        "lr": {"values": args.lr},
        "wd": {"values": args.wd},
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project=f"SVP Sweep scratch model={args.model}, num classes={args.num_classes}, cpr={args.compress_rate}",
)
args.output = os.path.join(
    args.output, args.model, str(args.num_classes), args.compress_rate
)


def main():
    accelerator = Accelerator(split_batches=True)
    run = wandb.init()
    wandb_batch_size = wandb.config.bs
    wandb_lr = wandb.config.lr
    wandb_weight_decay = wandb.config.wd
    accelerator.print(args)
    accelerator.print(
        f"batch_size={wandb_batch_size}, lr={wandb_lr}, weight_decay={wandb_weight_decay}"
    )

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    train_loader, test_loader = get_dataloaders(
        args.num_classes,
        args.dataset_path,
        args.mixup_alpha,
        args.cutmix_alpha,
        wandb_batch_size,
    )
    total_steps = args.epochs * len(train_loader)
    total_steps = 10 * (total_steps // 10)

    # Prepare model and parameter sets
    compress_rate = get_cpr(args.compress_rate)
    net = eval(args.model)(num_classes=args.num_classes, compress_rate=compress_rate)
    net, train_loader, test_loader = accelerator.prepare(net, train_loader, test_loader)

    # define criterion and aggregators
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    train_losses = []
    peak = 0
    best_state_dict = None

    start_time = 0
    epoch = 0
    target, idx_target = [i / 10 for i in range(1, 11)], 0

    net.train()
    last_print = 0
    for era in range(1 if args.no_warmup else 0, args.eras + 1):
        step, idx_target = 0, 0
        if accelerator.is_main_process:
            accelerator.print(
                "{:s}".format("Era " + str(era) if era > 0 else "Warming up")
            )

        # define optimizers/schedulers
        if era == 0:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=wandb_lr,
                momentum=args.momentum,
                weight_decay=wandb_weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=len(train_loader) * 5
            )
        else:
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=wandb_lr * (args.momentum ** (era - 1)),
                momentum=args.momentum,
                weight_decay=wandb_weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, total_steps, args.eta_min
            )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        total_steps_for_era = total_steps if era > 0 else 5 * len(train_loader)

        if start_time == 0:
            start_time = time.time()

        while step < total_steps_for_era:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                step += 1

                optimizer.zero_grad(set_to_none=True)
                outputs = net(inputs)

                loss = criterion(outputs, targets)

                accelerator.backward(loss)
                optimizer.step()

                train_losses.append(loss.item())
                train_losses = train_losses[-len(train_loader) :]

                scheduler.step()
                lr = scheduler.get_last_lr()[0]

                if time.time() - last_print > 0.05 or batch_idx + 1 == len(
                    train_loader
                ):
                    accelerator.print(
                        "\r{:6.2f}% loss:{:.4e} lr:{:.3e}".format(
                            100 * step / total_steps_for_era,
                            torch.mean(torch.tensor(train_losses)).item(),
                            lr,
                        ),
                        end="",
                    )
                    last_print = time.time()

                step_time = (time.time() - start_time) / (
                    total_steps * (era - 1 if era > 0 else 0)
                    + step
                    + (5 * len(train_loader) if era > 0 else 0)
                )
                remaining_time = (
                    total_steps_for_era - step + (args.eras - era) * total_steps
                ) * step_time

            score = test(net, test_loader) * 100
            accelerator.print(
                "{:6.2f}%".format(score),
                end="",
            )
            wandb.log(
                {
                    "peak": peak,
                    "top1": score,
                    "lr": lr,
                }
            )
            if score > peak:
                peak = score
                best_state_dict = copy.deepcopy(net.state_dict())
            accelerator.print(
                " {:4d}h{:02d}m epoch {:4d}".format(
                    int(remaining_time / 3600),
                    (int(remaining_time) % 3600) // 60,
                    epoch + 1,
                ),
                end="",
            )
            delta = score - args.target
            if delta > 0 and delta < 0.5:
                t_path = os.path.join(
                    args.output, f"{args.model}_{args.compress_rate}_{score:.2f}.pt"
                )
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                    },
                    t_path,
                )
            epoch += 1
            if (era == 0 and step >= total_steps_for_era) or (
                era > 0 and step / total_steps_for_era > target[idx_target]
            ):
                accelerator.print()
                if era > 0:
                    idx_target += 1

    total_time = time.time() - start_time
    accelerator.print()
    accelerator.print(
        "total time is {:4d}h{:02d}m".format(
            int(total_time / 3600), (int(total_time) % 3600) // 60
        )
    )

    name = f"{args.compress_rate}_{wandb_batch_size}_{wandb_lr}_{wandb_weight_decay}"
    path = os.path.join(args.output, f"{args.model}_{name}_{peak:.2f}.pt")
    torch.save(
        {
            "state_dict": best_state_dict,
        },
        path,
    )


if __name__ == "__main__":
    wandb.agent(sweep_id, function=main)
