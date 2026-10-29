"""Training and learning rate sweep for S2 Discrete-Time Hazard Survival Model.
Trace:
  S2HazardConfig -> DiscreteTimeHazardNet(cfg) -> lr sweep -> final training -> test metrics

Run via:
    python -m ml_models.S2_hazard.train
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_models.common.utils import set_seed, print_model_summary, save_metrics, count_parameters
from ml_models.common.splits import split_sequence_data
from ml_models.common.data import load_raw_dataset, SequentialBenchmarkDataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.S2_hazard.config import S2HazardConfig
from ml_models.S2_hazard.model import DiscreteTimeHazardNet


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        feats = batch["features"]
        masks = batch["mask"]
        labels_haz = batch["label_hazard"]

        optimizer.zero_grad()
        out = model(feats, masks)
        loss = criterion(out["hazard_logits"], labels_haz)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_model(model, loader):
    model.eval()
    all_probs = []
    all_distress = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"]
            masks = batch["mask"]
            labels_distress = batch["label_distress"]

            out = model(feats, masks)
            all_probs.extend(out["distress_prob"].cpu().numpy().tolist())
            all_distress.extend(labels_distress.cpu().numpy().tolist())

    return calculate_classification_metrics(all_distress, all_probs)


def run_s2_hazard():
    # 1. Config (Single source of truth)
    cfg = S2HazardConfig()
    set_seed(cfg.seed)

    # 2. Sequential dataset partition
    features, ages, masks, distress, hazard = load_raw_dataset()
    splits = split_sequence_data(distress, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)

    ds_train = SequentialBenchmarkDataset(features[splits.train_idx], ages[splits.train_idx], masks[splits.train_idx], distress[splits.train_idx], hazard[splits.train_idx])
    ds_val = SequentialBenchmarkDataset(features[splits.val_idx], ages[splits.val_idx], masks[splits.val_idx], distress[splits.val_idx], hazard[splits.val_idx])
    ds_test = SequentialBenchmarkDataset(features[splits.test_idx], ages[splits.test_idx], masks[splits.test_idx], distress[splits.test_idx], hazard[splits.test_idx])

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    best_lr = cfg.param_grid_lr[0]
    best_val_score = -1.0
    tuning_log = []

    print("[S2_hazard] Sweeping learning rates...")
    for lr in cfg.param_grid_lr:
        model = DiscreteTimeHazardNet(cfg=cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

        for _ in range(cfg.epochs):
            train_epoch(model, loader_train, optimizer, criterion)

        val_metrics = evaluate_model(model, loader_val)
        val_score = val_metrics["pr_auc"] if isinstance(val_metrics["pr_auc"], (int, float)) else 0.0
        tuning_log.append({"lr": lr, "val_pr_auc": val_score})
        print(f"  lr={lr:<8} | Val PR-AUC={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_lr = lr

    print(f"[S2_hazard] Selected optimal lr: {best_lr} (Val PR-AUC={best_val_score:.4f})")

    # 3. Final model trained with best lr
    final_model = DiscreteTimeHazardNet(cfg=cfg)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        train_epoch(final_model, loader_train, optimizer, criterion)

    test_metrics = evaluate_model(final_model, loader_test)
    test_metrics["best_lr"] = best_lr
    test_metrics["tuning_history"] = tuning_log

    params = count_parameters(final_model)
    print_model_summary(cfg.model_name, test_metrics, params=params)
    save_metrics("ml_models/S2_hazard/metrics.json", test_metrics)
    return test_metrics


if __name__ == "__main__":
    run_s2_hazard()
