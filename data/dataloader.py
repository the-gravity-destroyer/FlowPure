import os
import random
from collections import defaultdict

import lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from functools import partial


def stratified_indices_split(labels, val_ratio=0.1, seed=42):
    """Divides indices per class so that even rare classes appear in both splits.
    Classes with only 1 sample end up completely in the training set."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    train_indices, val_indices = [], []
    for indices in by_class.values():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        val_n = 0 if n == 1 else max(1, round(n * val_ratio))
        val_indices.extend(indices[:val_n])
        train_indices.extend(indices[val_n:])
    return train_indices, val_indices


class BaseDataModule(pl.LightningDataModule):
    """Images in [0, 1], integer labels. Shared by flow matching and classifiers."""

    AUGMENT = False
    CLASSIFIER_TYPES = {"resnet", "densenet"}

    def __init__(self, data_dir, batch_size, num_workers, model_type, dimensions, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.transform = self._get_transforms(dimensions, train = False)
        self.train_transform = self._get_transforms(dimensions, train=True)

    def _get_transforms(self, dimensions, train):
        if dimensions not in (32, 64):
            raise ValueError(f"Unsupported dimensions: {dimensions}")
        tf = [transforms.Lambda(lambda img: img.convert("RGB")),
              transforms.Resize((dimensions, dimensions))]
        if train and self.AUGMENT and self.hparams.model_type in self.CLASSIFIER_TYPES:
            tf += [transforms.RandomCrop(dimensions, padding=dimensions // 8),
                   transforms.RandomHorizontalFlip()]
        tf.append(transforms.ToTensor())
        return transforms.Compose(tf)

    def _random_split(self, dataset, val_ratio=0.1):
        val_len = int(len(dataset) * val_ratio)
        train_len = len(dataset) - val_len
        return random_split(dataset, [train_len, val_len],
                            generator=torch.Generator().manual_seed(42))

    @staticmethod
    def _split_indices(n, val_ratio=0.1):
        # identisch zu random_split(..., generator=manual_seed(42))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
        train_len = n - int(n * val_ratio)
        return perm[:train_len], perm[train_len:]
    
    @staticmethod
    def _collate(batch):
        if isinstance(batch[0], dict):  # Hugging Face datasets
            images = torch.stack([item["image"] for item in batch], dim=0)
            labels = torch.stack([item["label"] for item in batch], dim=0).long()
        else:  # torchvision datasets
            images = torch.stack([item[0] for item in batch], dim=0)
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

    def _get_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=self._collate,
            multiprocessing_context="fork" if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()


class HFDataModule(BaseDataModule):
    """Shared logic for datasets loaded from the Hugging Face Hub."""

    HF_NAME = None
    HF_SUBSET = None
    TRAIN_SPLIT = "train"
    TEST_SPLIT = None

    def _cache_dir(self):
        return os.path.join(self.hparams.data_dir, self.__class__.__name__.lower())

    def _load(self, split, transform):
        dataset = load_dataset(self.HF_NAME, self.HF_SUBSET, split=split,
                               cache_dir=self._cache_dir())
        dataset.set_transform(partial(self._apply_transforms, transform=transform))
        return dataset

    def _apply_transforms(self, batch, transform):
        batch["image"] = [transform(img) for img in batch["image"]]
        batch["label"] = [torch.tensor(l, dtype=torch.long) for l in batch["label"]]
        return batch

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            train_full = self._load(self.TRAIN_SPLIT, self.train_transform)
            val_full = self._load(self.TRAIN_SPLIT, self.transform)
            train_idx, val_idx = self._split_indices(len(train_full))
            self.train_dataset = Subset(train_full, train_idx)
            self.val_dataset = Subset(val_full, val_idx)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._load(self.TEST_SPLIT, self.transform)


class Cifar10DataModule(BaseDataModule):
    AUGMENT = True
    def __init__(self, data_dir, batch_size=64, num_workers=4, model_type=None):
        super().__init__(data_dir, batch_size, num_workers, model_type=model_type,
                         dimensions=32, num_classes=10)

    def prepare_data(self):
        datasets.CIFAR10(root=self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            root = self.hparams.data_dir
            train_full = datasets.CIFAR10(root=root, train=True, transform=self.train_transform)
            val_full = datasets.CIFAR10(root=root, train=True, transform=self.transform)
            train_idx, val_idx = self._split_indices(len(train_full))
            self.train_dataset = Subset(train_full, train_idx)
            self.val_dataset = Subset(val_full, val_idx)
        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.CIFAR10(root=self.hparams.data_dir, train=False,
                                                 transform=self.transform)


class TinyImageNetDataModule(HFDataModule):
    AUGMENT = True
    # Official test labels are not public -> the "valid" split serves as test set.
    HF_NAME = "zh-plus/tiny-imagenet"
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "valid"

    def __init__(self, data_dir, batch_size=64, num_workers=4, model_type=None):
        super().__init__(data_dir, batch_size, num_workers, model_type=model_type,
                         dimensions=64, num_classes=200)


class SynsetDataModule(HFDataModule):
    HF_NAME = "FraunhoferIOSB/Synset-Signset-Germany"
    TRAIN_SPLIT = "train"
    TEST_SPLIT = "validation"

    def __init__(self, data_dir, batch_size=64, num_workers=4, model_type="classifier", subset="OGRE"):
        self.HF_SUBSET = subset
        super().__init__(data_dir, batch_size, num_workers, model_type=model_type,
                         dimensions=32, num_classes=211)


class GTSRBDataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4, model_type="classifier"):
        super().__init__(data_dir, batch_size, num_workers, model_type=model_type,
                         dimensions=32, num_classes=43)

    def prepare_data(self):
        datasets.GTSRB(root=self.hparams.data_dir, split="train", download=True)
        datasets.GTSRB(root=self.hparams.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            full = datasets.GTSRB(root=self.hparams.data_dir, split="train", transform=self.transform)
            # TODO: split by track instead of by image (see notes)
            self.train_dataset, self.val_dataset = self._random_split(full)
        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.GTSRB(root=self.hparams.data_dir, split="test",
                                               transform=self.transform)


class DataModuleFactory:
    _REGISTRY = {
        "cifar10": Cifar10DataModule,
        "tiny_imagenet": TinyImageNetDataModule,
        "synset": SynsetDataModule,
        "gtsrb": GTSRBDataModule,
    }

    @staticmethod
    def create(dataset_name, model_type, data_dir, batch_size=64, num_workers=4):
        dataset_name = dataset_name.lower()
        if dataset_name not in DataModuleFactory._REGISTRY:
            raise ValueError(f"Dataset '{dataset_name}' not found. "
                             f"Available: {list(DataModuleFactory._REGISTRY.keys())}")
        cls = DataModuleFactory._REGISTRY[dataset_name]
        return cls(data_dir=data_dir, batch_size=batch_size,
                   num_workers=num_workers, model_type=model_type)