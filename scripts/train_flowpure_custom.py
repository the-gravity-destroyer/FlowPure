"""FlowPure_PGD-Training (Collaert et al., 2025) für eigene Datensätze und Klassifikatoren.

Trainingsverfahren identisch zu trainer_flowpure.py (--noise_type pgd):
  - x0 = PGD-Adversarial (10 Steps, alpha = 2/255, Random Start), eps ~ U(0, 0.05) pro Bild,
    on-the-fly in jeder Iteration gegen den Opfer-Klassifikator
  - x1 = sauberes Bild, OT-Pfad ohne Rauschen (ConditionalFlowMatcher(sigma=0))
  - NCSN++-Backbone aus diffusion_configs/cifar10.yml, Adam lr 2e-4, Batch 64, 300k Steps

Geändert gegenüber dem Original:
  - Daten: DataModuleFactory (Train-Split ohne Augmentation, Bilder in [0, 1])
  - Opfer: eigener Classifier-Checkpoint (Normalisierung steckt im Modell)
  - 64x64 (TinyImageNet): gleiche Config, nur data.image_size = 64
  - Checkpoint-Name <dataset>_<arch>, Resume-Option

Aufruf (aus dem FlowPure-Root; kein PYTHONPATH nötig, PROJECT_ROOT unten anpassen):
  python scripts/train_flowpure_custom.py --dataset cifar10 \
      --classifier /pfad/zu/resnet.ckpt/<timestamp>/last.ckpt --data_dir /home/leonis/project_datasets
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = "/home/leonis/projects/fm_attack_detector/fm_attack_detector"
FLOWPURE_ROOT = Path(__file__).resolve().parent
if FLOWPURE_ROOT.name == "scripts":  # Skript liegt in FlowPure/scripts/
    FLOWPURE_ROOT = FLOWPURE_ROOT.parent

# Nur FlowPure in den Suchpfad. Dein Projekt darf NICHT in sys.path: FlowPures `models/` ist ein
# Namespace-Paket (ohne __init__.py), dein `models/` ein reguläres -> deins würde immer gewinnen.
sys.path.insert(0, str(FLOWPURE_ROOT))
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(PROJECT_ROOT).resolve()]

import torch
import torchcfm

import attacks.batch_attacks as attacks
import score_sde.models.utils as mutils
from utils import RevDiff


def _alias_package(alias: str, pkg_dir: Path):
    """Registriert einen Ordner deines Projekts unter eigenem Namen (relative Imports funktionieren)."""
    pkg = types.ModuleType(alias)
    pkg.__path__ = [str(pkg_dir)]
    sys.modules[alias] = pkg


_alias_package("project_classifier", Path(PROJECT_ROOT) / "models" / "classifier")
_alias_package("project_data", Path(PROJECT_ROOT) / "data")
Classifier = importlib.import_module("project_classifier.classifier").Classifier
DataModuleFactory = importlib.import_module("project_data.dataloader").DataModuleFactory

IMAGE_SIZE = {"cifar10": 32, "gtsrb": 32, "tiny_imagenet": 64}


def build_flow_model(image_size: int, device: torch.device) -> RevDiff:
    """Wie utils.get_model('new'), aber mit wählbarer Bildgröße."""
    config = mutils.parse_config("diffusion_configs/cifar10.yml")
    config.data.image_size = image_size
    model = mutils.create_model(config).to(device)  # DataParallel(NCSN++)
    return RevDiff(model)


def load_victim(path: str, device: torch.device):
    victim = Classifier.load_from_checkpoint(path, map_location=device).to(device).eval()
    victim.requires_grad_(False)
    arch = victim.config["model"]["type"]  # "resnet" / "densenet"
    return victim, arch


def build_loader(dataset: str, data_dir: str, batch_size: int, num_workers: int):
    # "flow_matching" -> keine Augmentation (wie ToTensor() im Original)
    dm = DataModuleFactory.create(dataset, model_type="flow_matching", data_dir=data_dir,
                                  batch_size=batch_size, num_workers=num_workers)
    dm.prepare_data()
    dm.setup("fit")
    print(f"[DATA] {dataset}: {len(dm.train_dataset)} Trainingsbilder")
    return dm.train_dataloader()


def save(path: str, net_model: RevDiff, optim: torch.optim.Optimizer, step: int, meta: dict) -> None:
    # gleiche Struktur wie im Original ([model_state, optim_state]) plus Step und Metadaten
    torch.save([net_model.model.state_dict(), optim.state_dict(), step, meta], path)


def main() -> None:
    p = argparse.ArgumentParser(description="FlowPure_PGD-Training für eigene Datensätze")
    p.add_argument("--dataset", required=True, choices=list(IMAGE_SIZE))
    p.add_argument("--classifier", required=True, help="Pfad zum Classifier-Checkpoint (Opfer)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--max_steps", type=int, default=300001)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Effektive Batch Size (wie im Original 64)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Micro-Batches pro Optimizer-Step (bei OOM: 2 -> 2 x 32 = 64)")
    p.add_argument("--snapshot_freq", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="resources/checkpoints/flowpure_pgd")
    p.add_argument("--resume", default=None, help="Checkpoint zum Fortsetzen")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    victim, arch = load_victim(args.classifier, device)
    name = f"{args.dataset}_{arch}"
    meta = {"dataset": args.dataset, "arch": arch, "classifier": args.classifier,
            "image_size": IMAGE_SIZE[args.dataset]}

    if args.batch_size % args.grad_accum:
        raise ValueError("--batch_size muss durch --grad_accum teilbar sein")
    micro_bs = args.batch_size // args.grad_accum
    loader = build_loader(args.dataset, args.data_dir, micro_bs, args.num_workers)
    net_model = build_flow_model(IMAGE_SIZE[args.dataset], device)
    atk = attacks.PGD(victim)  # eps wird pro Batch überschrieben, alpha = 2/255, steps = 10
    FM = torchcfm.ConditionalFlowMatcher(sigma=0)
    optim = torch.optim.Adam(net_model.parameters(), lr=2e-4)

    step = 0
    if args.resume:
        states = torch.load(args.resume, map_location=device)
        net_model.model.load_state_dict(states[0])
        optim.load_state_dict(states[1])
        step = states[2] + 1
        print(f"[RESUME] {args.resume}, weiter bei Step {step}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[TRAIN] {name}, {sum(q.numel() for q in net_model.parameters()) / 1e6:.1f}M Parameter, "
          f"Batch {args.batch_size} = {args.grad_accum} x {micro_bs}")

    batches = iter(loader)
    start_step = last_step = step
    t_last = time.time()

    def next_batch():
        nonlocal batches
        try:
            return next(batches)
        except StopIteration:  # neue Epoche
            batches = iter(loader)
            return next(batches)

    while step <= args.max_steps:
        net_model.train()
        optim.zero_grad()
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            X, y = next_batch()
            x1 = X.to(device)

            # wie im Original: eps ~ U(0, 0.05) pro Bild
            epsilon = (torch.rand(y.size()) * 0.05).view(y.size(0), 1, 1, 1).to(device)
            x0 = atk(X, y, eps=epsilon).clone().detach()

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2) / args.grad_accum
            loss.backward()
            loss_sum += loss.item()
        optim.step()

        if step % 100 == 0:
            now = time.time()
            if step > start_step:
                s_per_step = (now - t_last) / (step - last_step)
                eta_h = s_per_step * (args.max_steps - step) / 3600
                print(f"step: {step}/{args.max_steps}, loss: {loss_sum:.6f}, "
                      f"{s_per_step:.3f} s/step, Rest ~{eta_h:.1f} h ({eta_h / 24:.1f} Tage)", flush=True)
            else:
                print(f"step: {step}/{args.max_steps}, loss: {loss_sum:.6f}", flush=True)
            t_last, last_step = now, step

        if step % args.snapshot_freq == 0:
            save(os.path.join(args.out_dir, f"{name}_{step}.pth"), net_model, optim, step, meta)
            save(os.path.join(args.out_dir, f"{name}.pth"), net_model, optim, step, meta)
            print(f"[SAVE] {name} @ {step}")

        step += 1


if __name__ == "__main__":
    main()