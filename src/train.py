"""Training loop for the pricing MLP (Section 3.2-3.3 of the article).

Uses Adam optimizer with step-decay learning rate (Figure 4 of the article).

CRITICAL FIX vs previous implementation:
- Previously used `ExponentialLR` (smooth exponential decay).
- Article Figure 4 clearly shows a **step decay** (discontinuous drops) at
  epochs ~1000 and ~2000 with gamma=0.1, giving 1e-3 -> 1e-4 -> 1e-5.
- Fixed to use `MultiStepLR(milestones=[epochs//3, 2*epochs//3], gamma=0.1)`.
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=3000,
    lr_start=1e-3,
    lr_end=1e-5,
    schedule="decay",
    device=None,
    print_every=1,
    progress=True,
    log_prefix="",
):
    """Train the MLP model with Adam + configurable LR schedule.

    Args:
        model: PricingMLP instance
        train_loader: DataLoader for training data
        val_loader: optional DataLoader for validation
        epochs: number of training epochs (article: 3000)
        lr_start: initial learning rate (article: 1e-3)
        lr_end: final learning rate (article: 1e-5)
        schedule: "decay" (step decay, Fig 4), "constant", or "cyclical"
        device: torch device
        print_every: print loss every N epochs (default 1 = every epoch)
        progress: show tqdm progress bar over epochs
        log_prefix: optional string prefix on each per-epoch log line

    Returns:
        dict with 'train_losses', 'val_losses', 'lrs', 'epoch_times'
    """
    if device is None:
        device = next(model.parameters()).device

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)

    # LR schedule selection (Section 3.3, Figure 4 of the article)
    if schedule == "decay":
        # Step decay per article Figure 4: milestones at 1/3 and 2/3 of total epochs,
        # gamma=0.1 gives lr 1e-3 -> 1e-4 -> 1e-5.
        m1 = max(1, epochs // 3)
        m2 = max(m1 + 1, 2 * epochs // 3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[m1, m2], gamma=0.1
        )
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
    lrs = []
    epoch_times = []

    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"{log_prefix}training", total=epochs, ncols=100)

    for epoch in iterator:
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

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

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()
                    n_val += 1
            avg_val_loss = val_loss / n_val
            val_losses.append(avg_val_loss)

        if not step_per_batch and scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        epoch_times.append(time.time() - t0)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            ts = time.strftime("%H:%M:%S")
            msg = (f"  [{ts}] {log_prefix}Epoch {epoch:4d}/{epochs} | "
                   f"Train MSE: {avg_train_loss:.6e} | LR: {lr:.2e}")
            if avg_val_loss is not None:
                msg += f" | Val MSE: {avg_val_loss:.6e}"
            if progress:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses if val_loader else None,
        "lrs": lrs,
        "epoch_times": epoch_times,
    }


def lr_range_test(model, train_loader, lr_min=1e-9, lr_max=10, num_steps=200, device=None):
    """LR range test (Smith 2015, Section 3.3, Figure 3 of the article).

    Sweeps LR geometrically from lr_min to lr_max and records per-step loss.
    The whole range is traversed so the characteristic U-shape (flat, drop,
    rise) is fully visible — matching the article's Figure 3. The only early
    exit is on non-finite loss (NaN/Inf), after which remaining points are
    back-filled with the last finite value to keep arrays aligned.
    """
    import math

    if device is None:
        device = next(model.parameters()).device

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_min)

    mult = (lr_max / lr_min) ** (1.0 / num_steps)

    lrs = []
    losses = []
    best_loss = float("inf")

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

        lr = optimizer.param_groups[0]["lr"]
        loss_val = loss.item()
        lrs.append(lr)
        losses.append(loss_val if math.isfinite(loss_val) else best_loss * 10)

        if math.isfinite(loss_val) and loss_val < best_loss:
            best_loss = loss_val
        # Stop once the rise is clearly established — avoids Adam's post-divergence
        # oscillations (ReLU saturation then reactivation on new batches).
        if (not math.isfinite(loss_val) or loss_val > 50 * best_loss) and i > 20:
            break

        loss.backward()
        optimizer.step()

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
