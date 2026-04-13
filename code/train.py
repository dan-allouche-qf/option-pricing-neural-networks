"""Training loop for the pricing MLP (Section 3.2-3.3 of the article).

Uses Adam optimizer with decaying learning rate schedule (10^-3 -> 10^-5).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=3000,
    lr_start=1e-3,
    lr_end=1e-5,
    schedule="decay",
    device=None,
    print_every=100,
):
    """Train the MLP model with Adam + configurable LR schedule.

    Args:
        model: PricingMLP instance
        train_loader: DataLoader for training data
        val_loader: optional DataLoader for validation
        epochs: number of training epochs (article: 3000)
        lr_start: initial learning rate (article: 1e-3)
        lr_end: final learning rate (article: 1e-5)
        schedule: "decay" (default), "constant", or "cyclical" (Section 3.3, Figure 4)
        device: torch device
        print_every: print loss every N epochs

    Returns:
        dict with 'train_losses' and 'val_losses' (list of floats per epoch)
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    # LR schedule selection (Section 3.3, Figure 4 of the article)
    if schedule == "decay":
        # Exponential decay: lr_end = lr_start * gamma^epochs
        gamma = (lr_end / lr_start) ** (1.0 / epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        step_per_batch = False
    elif schedule == "constant":
        scheduler = None
        step_per_batch = False
    elif schedule == "cyclical":
        # Cyclical LR (Smith 2015, ref [29] in article)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr_end,
            max_lr=lr_start,
            step_size_up=500 * steps_per_epoch,
            mode="triangular",
            cycle_momentum=False,
        )
        step_per_batch = True
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            if step_per_batch and scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # --- Validation ---
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()
                    n_val += 1
            avg_val_loss = val_loss / n_val
            val_losses.append(avg_val_loss)

        if not step_per_batch and scheduler is not None:
            scheduler.step()

        if epoch % print_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            msg = f"  Epoch {epoch:4d}/{epochs} | Train MSE: {avg_train_loss:.6e} | LR: {lr:.2e}"
            if avg_val_loss is not None:
                msg += f" | Val MSE: {avg_val_loss:.6e}"
            print(msg)

    return {"train_losses": train_losses, "val_losses": val_losses if val_loader else None}


def lr_range_test(model, train_loader, lr_min=1e-9, lr_max=10, num_steps=200, device=None):
    """LR range test (Smith 2015, Section 3.3, Figure 3 of the article).

    Progressively increases learning rate and records loss to find optimal range.

    Returns:
        lrs: list of learning rates
        losses: list of corresponding average losses
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_min)

    # Exponential increase: lr_max = lr_min * mult^num_steps
    mult = (lr_max / lr_min) ** (1.0 / num_steps)

    lrs = []
    losses = []
    best_loss = float("inf")
    step = 0

    model.train()
    data_iter = iter(train_loader)

    for i in range(num_steps):
        try:
            X_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            X_batch, y_batch = next(data_iter)

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        losses.append(loss.item())

        # Stop if loss diverges (> 4x best)
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 4 * best_loss and i > 10:
            break

        # Increase LR
        for param_group in optimizer.param_groups:
            param_group["lr"] *= mult

    return lrs, losses


@torch.no_grad()
def predict(model, X, device=None, batch_size=8192):
    """Run inference on numpy array X, return numpy array."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device)
        preds.append(model(batch).cpu())

    return torch.cat(preds, dim=0).numpy()
