import torch
import time
import copy

def mae_score(true, pred):
    return torch.mean(torch.abs(true - pred))

def rmse_score(true, pred):
    return torch.sqrt(torch.mean((true - pred)**2))

def f1_score(true, pred):
    target = torch.where((true>0.05)&(true<0.5))
    true = true[target]
    pred = pred[target]

    true = torch.where(true<0.15, 0, 1)
    pred = torch.where(pred<0.15, 0, 1)

    right = torch.sum(true*pred == 1)
    precision = right / torch.sum(true + 1e-8)
    recall = right / torch.sum(pred + 1e-8)
    score = 2 * precision * recall / (precision + recall + 1e-8)
    return score

def mae_over_f1(true, pred):
    mae = mae_score(true, pred)
    f1 = f1_score(true, pred)
    score = mae / (f1 + 1e-8)
    loss = rmse_score(true, pred) - f1
    return mae, f1, score, loss

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    losses = {"mae": 0, "f1": 0, "mae_over_f1": 0}

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for feature, target, mask in data_loader:
        feature = feature.float().to(device)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(feature)
        mae, f1, score, loss = mae_over_f1(target, pred)
        loss.backward()
        optimizer.step()

        losses["mae"] += mae.item()
        losses["f1"] += f1.item()
        losses["mae_over_f1"] += score.item()

        if lr_scheduler is not None:
            lr_scheduler.step()

    losses["mae"] /= len(data_loader)
    losses["f1"] /= len(data_loader)
    losses["mae_over_f1"] /= len(data_loader)
    return losses

@torch.no_grad()
def evalutate(model, data_loader, device):
    model.eval()
    losses = {"mae": 0, "f1": 0, "mae_over_f1": 0}

    for feature, target, mask in data_loader:
        feature = feature.float().to(device)
        target = target.to(device)

        pred = model(feature)
        mae, f1, score, loss = mae_over_f1(target, pred)

        losses["mae"] += mae.item()
        losses["f1"] += f1.item()
        losses["mae_over_f1"] += score.item()

    losses["mae"] /= len(data_loader)
    losses["f1"] /= len(data_loader)
    losses["mae_over_f1"] /= len(data_loader)
    return losses

def train_model(model, optimizer, scheduler, device, save_dir, start_epoch, end_epoch,
                train_dataloader, valid_dataloader):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.

    train_losses = {"mae": [], "f1": [], "mae_over_f1": []}
    valid_losses = {"mae": [], "f1": [], "mae_over_f1": []}

    for epoch in range(start_epoch, end_epoch+1):
        print("\n" + "="*40)
        print("Eopch {}/{}".format(epoch, end_epoch))

        train_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch)
        train_losses["mae"].append(train_loss["mae"])
        train_losses["f1"].append(train_loss["f1"])
        train_losses["mae_over_f1"].append(train_loss["mae_over_f1"])
        print("\nTrain Loss")
        print("mae: {:.6f}\t f1: {:.6f}\t mae_over_f1: {:.6f}".format(train_loss["mae"],
                                                          train_loss["f1"],
                                                          train_loss["mae_over_f1"]))
        if scheduler is not None:
            scheduler.step()

        valid_loss = evalutate(model, valid_dataloader, device)
        valid_losses["mae"].append(valid_loss["mae"])
        valid_losses["f1"].append(valid_loss["f1"])
        valid_losses["mae_over_f1"].append(valid_loss["mae_over_f1"])
        print("\nValid Loss")
        print("mae: {:.6f}\t f1: {:.6f}\t mae_over_f1: {:.6f}".format(valid_loss["mae"],
                                                          valid_loss["f1"],
                                                          valid_loss["mae_over_f1"]))
        if valid_loss["mae_over_f1"] < best_loss:
            best_loss = valid_loss["mae_over_f1"]
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("\nTraining complete in {}m {:0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:.4f}".format(best_loss))

    torch.save(best_model_wts, save_dir)
    return train_losses, valid_losses