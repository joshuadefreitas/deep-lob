import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_lob.dataset import DeepLOBDataset
from deep_lob.models import DeepLOBModel


def compute_confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    num_classes = 3  # down, flat, up (mapped to 0,1,2)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # overall accuracy
    accuracy = float((y_true == y_pred).mean())

    per_class = {}
    f1s = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[c, :].sum()),
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    return cm, metrics


def plot_confusion_matrix(cm: np.ndarray, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping confusion matrix plot.")
        return

    class_names = ["down", "flat", "up"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_model(data_path: Path, model_path: Path, batch_size: int = 128, device_str: str = "auto"):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # dataset and loader
    dataset = DeepLOBDataset(str(data_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # infer number of features from first sample
    x0, _ = dataset[0]
    num_features = x0.shape[1]

    # model
    model = DeepLOBModel(num_features=num_features)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    cm, metrics = compute_confusion_and_metrics(y_true, y_pred)

    # print summary
    print("\n=== Evaluation Summary ===")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Macro F1   : {metrics['macro_f1']:.4f}")
    print("Per class:")
    for c, stats in metrics["per_class"].items():
        print(
            f"  Class {c} -> "
            f"P={stats['precision']:.3f}, "
            f"R={stats['recall']:.3f}, "
            f"F1={stats['f1']:.3f}, "
            f"support={stats['support']}"
        )

    # save metrics + plot
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    cm_path = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)
    print(f"Saved confusion matrix plot to {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepLOB model on a dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/lob_windows.npz",
        help="Path to NPZ file with X and y.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/deeplob_synthetic.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'.",
    )

    args = parser.parse_args()
    evaluate_model(Path(args.data), Path(args.model), batch_size=args.batch_size, device_str=args.device)


if __name__ == "__main__":
    main()