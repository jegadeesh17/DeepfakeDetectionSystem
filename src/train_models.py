import copy
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data_preprocessing import get_dataloaders
from src.evaluation import evaluate_model
from src.model_builder import build_efficientnet, get_device, set_backbone_trainable
from src.model_loader import save_checkpoint


def _ensure_torch_dynamo() -> None:
    """Recover from partial torch._dynamo imports after interrupted notebook runs."""
    dynamo = sys.modules.get("torch._dynamo")
    if dynamo is not None and not hasattr(dynamo, "decorators"):
        for key in list(sys.modules):
            if key == "torch._dynamo" or key.startswith("torch._dynamo."):
                del sys.modules[key]
    import torch._dynamo.decorators  # noqa: F401


def train_model(
    model,
    train_loader,
    val_loader,
    model_name,
    device=None,
    epochs=5,
    patience=3,
    lr=1e-4,
    architecture="EfficientNet",
    gradient_accumulation_steps=1,
):
    if device is None:
        device = get_device()

    model = model.to(device)

    df = train_loader.dataset.dataframe
    num_pos = (df["label"].str.upper() == "FAKE").sum()
    num_neg = (df["label"].str.upper() == "REAL").sum()
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    _ensure_torch_dynamo()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(
        f"\n--- Starting training for {model_name} with lr={lr}, "
        f"grad_accum={gradient_accumulation_steps} ---"
    )
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, (inputs, labels) in enumerate(train_bar, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels.float()) / gradient_accumulation_steps
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                corrects = torch.sum(preds == labels.float())

            scaler.scale(loss).backward()

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
            train_corrects += corrects.item()
            train_bar.set_postfix(
                {"loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                 "acc": f"{corrects.item() / inputs.size(0):.4f}"}
            )

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, labels.float())
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    corrects = torch.sum(preds == labels.float())

                val_loss += loss.item() * inputs.size(0)
                val_corrects += corrects.item()
                val_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{corrects.item() / inputs.size(0):.4f}"}
                )

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects / len(val_loader.dataset)

        print(
            f"{model_name} - Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered for {model_name} at epoch {epoch + 1}.")
                break

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    model.load_state_dict(best_model_wts)
    return model, best_val_loss


def run_training_pipeline(
    batch_size=8,
    epochs=20,
    patience=5,
    lr=1e-3,
    gradient_accumulation_steps=4,
    force_rebuild=False,
):
    """Train frozen-backbone EfficientNet (1650 Ti safe) and evaluate on held-out test set."""
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, force_rebuild=force_rebuild
    )
    print(
        f"Split sizes — train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )

    architecture = "EfficientNet"
    model = build_efficientnet()
    set_backbone_trainable(model, architecture, trainable=False)

    trained_model, _ = train_model(
        model,
        train_loader,
        val_loader,
        architecture,
        device=device,
        epochs=epochs,
        patience=patience,
        lr=lr,
        architecture=architecture,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    print("\n--- Validation set ---")
    evaluate_model(trained_model, val_loader, architecture, device=device)

    print("\n--- Held-out test set ---")
    evaluate_model(trained_model, test_loader, f"{architecture} (test)", device=device)

    save_path = save_checkpoint(trained_model, architecture)
    print(f"Final model saved to {save_path}")


if __name__ == "__main__":
    run_training_pipeline()
