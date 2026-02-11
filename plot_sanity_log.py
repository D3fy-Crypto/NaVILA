#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


def parse_log(text):
    loss_pattern = re.compile(r"'loss'\s*:\s*([0-9.]+)")
    lr_pattern = re.compile(r"'learning_rate'\s*:\s*([0-9.eE+-]+)")
    grad_pattern = re.compile(r"'grad_norm/([^']+)'\s*:\s*([0-9.eE+-]+)")

    losses = []
    lrs = []
    grad_norms = {}

    for line in text.splitlines():
        loss_match = loss_pattern.search(line)
        lr_match = lr_pattern.search(line)
        if loss_match:
            losses.append(float(loss_match.group(1)))
            if lr_match:
                lrs.append(float(lr_match.group(1)))
            else:
                lrs.append(None)

        grad_matches = grad_pattern.findall(line)
        if grad_matches:
            step = len(losses) - 1 if losses else None
            for name, value in grad_matches:
                grad_norms.setdefault(name, []).append((step, float(value)))

    return losses, lrs, grad_norms


def write_csv(output_path, losses, lrs):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "learning_rate"])
        for i, loss in enumerate(losses):
            writer.writerow([i, loss, lrs[i] if i < len(lrs) else None])


def plot_curves(output_dir, losses, lrs, grad_norms):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available: {exc}")
        return False

    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(losses)), losses, label="loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close()

    if lrs and any(lr is not None for lr in lrs):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(lrs)), [lr if lr is not None else float("nan") for lr in lrs], label="learning_rate")
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title("Learning Rate")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "lr_curve.png", dpi=150)
        plt.close()

    if grad_norms:
        plt.figure(figsize=(10, 5))
        for name, series in grad_norms.items():
            if not series:
                continue
            xs, ys = zip(*[(s, v) for s, v in series if s is not None])
            plt.plot(xs, ys, label=name)
        plt.xlabel("step")
        plt.ylabel("grad norm")
        plt.title("Gradient Norms")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "grad_norms.png", dpi=150)
        plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Plot sanity check logs")
    parser.add_argument("log_file", help="Path to training log")
    parser.add_argument("--out", default=None, help="Output directory for plots")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir = Path(args.out) if args.out else log_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text()
    losses, lrs, grad_norms = parse_log(text)

    print(f"Parsed {len(losses)} loss entries")
    if losses:
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Loss change: {losses[0] - losses[-1]:.4f}")

    write_csv(output_dir / "loss_lr.csv", losses, lrs)
    if plot_curves(output_dir, losses, lrs, grad_norms):
        print(f"Saved plots to {output_dir}")
    else:
        print("Skipped plots due to missing matplotlib; CSV still written.")


if __name__ == "__main__":
    main()
