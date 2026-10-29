"""Training and fine-tuning pipeline for M7 Pre-trained Sequence Model.
Trace:
  M7PretrainedConfig -> Phase 1 SSL Pre-training (Masked Reconstruction) -> Phase 2 Fine-Tuning Sweep -> test metrics

Run via:
    python -m ml_models.M7_pretrained.train
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml_models.common.utils import set_seed, print_model_summary, save_metrics, count_parameters
from ml_models.common.splits import split_sequence_data
from ml_models.common.data import load_raw_dataset, SequentialBenchmarkDataset
from ml_models.common.metrics import calculate_classification_metrics
from ml_models.M7_pretrained.config import M7PretrainedConfig
from ml_models.M7_pretrained.model import PretrainedSequenceBackbone


def pretrain_backbone(model, loader, epochs=10, lr=1e-3, mask_ratio=0.15):
    """Self-supervised pre-training via masked sequence reconstruction."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()

    print("[M7_pretrained] Phase 1: Self-supervised pretraining (masked reconstruction)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            feats = batch["features"]  # [B, 12, 44]
            # Random masking: mask configured percentage of sequence values
            noise_mask = (torch.rand_like(feats) > mask_ratio).float()
            corrupted = feats * noise_mask

            optimizer.zero_grad()
            recon = model.forward_pretrain(corrupted)
            loss = criterion(recon, feats)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Pretrain Epoch {epoch+1}/{epochs} | MSE Loss: {epoch_loss / max(1, len(loader)):.4f}")


def finetune_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        feats = batch["features"]
        labels = batch["label_distress"]

        optimizer.zero_grad()
        logits = model.forward_finetune(feats)
        loss = criterion(logits, labels)
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
            labels = batch["label_distress"]
            logits = model.forward_finetune(feats)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_distress.extend(labels.cpu().numpy().tolist())
    return calculate_classification_metrics(all_distress, all_probs)


def run_m7_pretrained():
    # 1. Config (Single source of truth)
    cfg = M7PretrainedConfig()
    set_seed(cfg.seed)

    # 2. Sequential dataset partition
    features, ages, masks, distress, hazard = load_raw_dataset()
    splits = split_sequence_data(distress, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.seed)

    ds_train = SequentialBenchmarkDataset(features[splits.train_idx], ages[splits.train_idx], masks[splits.train_idx], distress[splits.train_idx])
    ds_val = SequentialBenchmarkDataset(features[splits.val_idx], ages[splits.val_idx], masks[splits.val_idx], distress[splits.val_idx])
    ds_test = SequentialBenchmarkDataset(features[splits.test_idx], ages[splits.test_idx], masks[splits.test_idx], distress[splits.test_idx])

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False)

    # 3. Base Self-Supervised Pre-training
    base_model = PretrainedSequenceBackbone(cfg=cfg)
    pretrain_backbone(base_model, loader_train, epochs=cfg.pretrain_epochs, lr=cfg.pretrain_lr, mask_ratio=cfg.mask_ratio)

    # 4. Downstream Fine-Tuning Hyperparameter Sweep
    criterion = nn.BCEWithLogitsLoss()
    best_lr = cfg.param_grid_lr[0]
    best_val_score = -1.0
    tuning_log = []

    print("[M7_pretrained] Phase 2: Downstream fine-tuning learning rate sweep...")
    for lr in cfg.param_grid_lr:
        model = copy.deepcopy(base_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

        for _ in range(cfg.finetune_epochs):
            finetune_epoch(model, loader_train, optimizer, criterion)

        val_metrics = evaluate_model(model, loader_val)
        val_score = val_metrics["pr_auc"] if isinstance(val_metrics["pr_auc"], (int, float)) else 0.0
        tuning_log.append({"lr": lr, "val_pr_auc": val_score})
        print(f"  finetune_lr={lr:<8} | Val PR-AUC={val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_lr = lr

    print(f"[M7_pretrained] Selected optimal finetune lr: {best_lr} (Val PR-AUC={best_val_score:.4f})")

    # 5. Final model trained with optimal finetune lr
    final_model = copy.deepcopy(base_model)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_lr, weight_decay=cfg.weight_decay)
    for _ in range(cfg.finetune_epochs):
        finetune_epoch(final_model, loader_train, optimizer, criterion)

    test_metrics = evaluate_model(final_model, loader_test)
    test_metrics["best_finetune_lr"] = best_lr
    test_metrics["tuning_history"] = tuning_log

    params = count_parameters(final_model)
    print_model_summary(cfg.model_name, test_metrics, params=params)
    save_metrics("ml_models/M7_pretrained/metrics.json", test_metrics)
    return test_metrics


if __name__ == "__main__":
    run_m7_pretrained()
