#!/usr/bin/env python3
"""
Training script for Quantum Circuit Heterogeneous Graph Transformer (QCHGT).

This script trains and evaluates the QCHGT model on the threshold class prediction task.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn import (
    create_hetero_gnn_model,
    create_threshold_class_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
)


def compute_challenge_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute challenge scoring metric (penalizes underprediction heavily)."""
    scores = []
    for pred, target in zip(predictions, targets):
        if pred < target:
            scores.append(0.0)
        else:
            rung_distance = pred - target
            scores.append(max(0.0, 1.0 - 0.1 * rung_distance))
    return np.mean(scores)


def train_epoch(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )
        
        loss = loss_fn(logits, batch.threshold_class)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.threshold_class.shape[0]
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch.threshold_class).sum().item()
        total_samples += batch.threshold_class.shape[0]
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, val_loader, device, loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in val_loader:
        batch = batch.to(device)
        
        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )
        
        loss = loss_fn(logits, batch.threshold_class)
        total_loss += loss.item() * batch.threshold_class.shape[0]
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch.threshold_class.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = (all_preds == all_targets).mean()
    challenge_score = compute_challenge_score(all_preds, all_targets)
    
    conservative_preds = np.minimum(all_preds + 1, len(THRESHOLD_LADDER) - 1)
    conservative_score = compute_challenge_score(conservative_preds, all_targets)
    
    return {
        "loss": total_loss / len(all_preds),
        "accuracy": accuracy,
        "challenge_score": challenge_score,
        "conservative_score": conservative_score,
        "predictions": all_preds,
        "targets": all_targets,
    }


class LabelSmoothingCE(nn.Module):
    """Label smoothing cross-entropy with conservative bias."""
    
    def __init__(self, num_classes: int = 9, smoothing: float = 0.1, upward_bias: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.upward_bias = upward_bias
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = targets.shape[0]
        device = logits.device
        
        smooth_labels = torch.full(
            (batch_size, self.num_classes),
            self.smoothing / self.num_classes,
            device=device
        )
        
        main_weight = 1.0 - self.smoothing - self.upward_bias
        for i in range(batch_size):
            t = targets[i].item()
            smooth_labels[i, t] = main_weight
            
            if t < self.num_classes - 1:
                remaining = self.upward_bias
                for k in range(t + 1, self.num_classes):
                    weight = remaining * 0.5
                    smooth_labels[i, k] += weight
                    remaining -= weight
        
        smooth_labels = smooth_labels / smooth_labels.sum(dim=1, keepdim=True)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        return loss.mean()


def main():
    parser = argparse.ArgumentParser(description="Train QCHGT model")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--upward-bias", type=float, default=0.1, help="Conservative bias")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-model", type=str, default=None, help="Path to save model")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    print("Loading data...")
    train_loader, val_loader = create_threshold_class_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=args.batch_size,
        val_fraction=0.2,
        seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("\nCreating QCHGT model...")
    model = create_hetero_gnn_model(
        model_type="threshold",
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    print(f"Model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    loss_fn = LabelSmoothingCE(
        num_classes=len(THRESHOLD_LADDER),
        smoothing=args.label_smoothing,
        upward_bias=args.upward_bias,
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    print("\nStarting training...")
    best_score = 0
    best_epoch = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_results = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()
        
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_results["loss"],
            "val_acc": val_results["accuracy"],
            "challenge_score": val_results["challenge_score"],
            "conservative_score": val_results["conservative_score"],
        }
        history.append(epoch_stats)
        
        is_best = val_results["challenge_score"] > best_score
        if is_best:
            best_score = val_results["challenge_score"]
            best_epoch = epoch
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)
        
        if epoch % 10 == 0 or epoch == 1 or is_best:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                f"Val Loss: {val_results['loss']:.4f} Acc: {val_results['accuracy']:.2%} | "
                f"Challenge: {val_results['challenge_score']:.4f} "
                f"{'*BEST*' if is_best else ''}"
            )
    
    print(f"\nBest challenge score: {best_score:.4f} at epoch {best_epoch}")
    
    final_results = evaluate(model, val_loader, device, loss_fn)
    print("\nFinal Results:")
    print(f"  Accuracy: {final_results['accuracy']:.2%}")
    print(f"  Challenge Score: {final_results['challenge_score']:.4f}")
    print(f"  Conservative Score: {final_results['conservative_score']:.4f}")
    
    preds = final_results["predictions"]
    targets = final_results["targets"]
    
    print("\nPer-class accuracy:")
    for i, threshold in enumerate(THRESHOLD_LADDER):
        mask = targets == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == i).mean()
            class_conservative = (preds[mask] >= i).mean()
            print(f"  Class {i} (threshold={threshold:3d}): "
                  f"Acc={class_acc:.2%}, Conservative={class_conservative:.2%}, N={mask.sum()}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = project_root / "results" / f"qchgt_training_{timestamp}.json"
    
    results = {
        "model": model.name,
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "upward_bias": args.upward_bias,
        },
        "best_score": best_score,
        "best_epoch": best_epoch,
        "final_accuracy": float(final_results["accuracy"]),
        "final_challenge_score": float(final_results["challenge_score"]),
        "final_conservative_score": float(final_results["conservative_score"]),
        "history": history,
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
