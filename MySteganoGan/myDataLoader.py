import numpy as np
import torch
import torchvision
from torchvision import transforms

DefaultTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

#-----有限datasets
class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)

    def __getitem__(self, index):
        max_attempts = 10  # 尝试次数，避免无限递归
        for _ in range(max_attempts):
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"\n Error loading image at index {index}: {e}. Skipping to next image.")
                index = (index + 1) % len(self)
        # 如果多次尝试仍然失败，则报错
        raise RuntimeError("Failed to load a valid image after several attempts.")

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=16, batch_size=4, *args, **kwargs):

        if transform is None:
            transform = DefaultTransform

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            *args,
            **kwargs
        )
