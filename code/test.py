import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from networks import NetworkA, NetworkB
from datasets.dataloader import get_datasets
from losses import supervised_loss, unsupervised_loss, ib_loss, cl_loss
from utils import AverageMeter, set_seed, adjust_learning_rate

# To protect our academic contributions, we have currently released only a simplified version of the code.
# The full code will be open-sourced after the paper is officially published.

def train(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train_labeled_loader, train_unlabeled_loader = get_datasets(args)


    model_a = NetworkA().to(device)
    model_b = NetworkB().to(device)


    optimizer = optim.SGD(list(model_a.parameters()) + list(model_b.parameters()),
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model_a.train()
        model_b.train()
        losses = AverageMeter()

        for (labeled_batch, unlabeled_batch) in zip(train_labeled_loader, train_unlabeled_loader):

            xs_rgb, hs_hsl, y_true = labeled_batch  # RGB, HSL, GT
            xu_rgb, hu_hsl = unlabeled_batch

            xs_rgb, hs_hsl, y_true = xs_rgb.to(device), hs_hsl.to(device), y_true.to(device)
            xu_rgb, hu_hsl = xu_rgb.to(device), hu_hsl.to(device)


            out_a_sup = model_a(xs_rgb)
            out_b_sup = model_b(hs_hsl)

            out_a_unsup = model_a(xu_rgb)
            out_b_unsup = model_b(hu_hsl)


            pseudo_b = (out_b_unsup > 0.5).float().detach()
            pseudo_a = (out_a_unsup > 0.5).float().detach()


            L_sup = supervised_loss(out_a_sup, y_true) + supervised_loss(out_b_sup, y_true)
            L_unsup = unsupervised_loss(out_a_unsup, pseudo_b) + unsupervised_loss(out_b_unsup, pseudo_a)
            L_ib = ib_loss(model_a, xs_rgb, y_true, xu_rgb, pseudo_b) + ib_loss(model_b, hs_hsl, y_true, hu_hsl, pseudo_a)
            L_cl = cl_loss(model_a, xs_rgb, y_true, xu_rgb, pseudo_b) + cl_loss(model_b, hs_hsl, y_true, hu_hsl, pseudo_a)


            loss = L_sup + args.alpha * L_ib + args.beta * L_cl + L_unsup


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {losses.avg:.4f}")
        adjust_learning_rate(optimizer, epoch, args)

    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)
