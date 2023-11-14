import os
import re
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics as tm
import torchvision as tv
from lightning.pytorch import callbacks

IMAGE_WIDTH, IMAGE_HEIGHT = 300, 300
NUM_EPOCHS = 100
NUM_WORKER = 16
NUM_CLASSES = 4
MONITOR = 'val_loss'
CLASSES = ['blank', 'paper', 'rock', 'scissors']


class RPSCNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),  # 20
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=5, padding=1),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),  # 32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(78400, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, NUM_CLASSES),
        )
        config = {'task': 'multiclass', 'num_classes': NUM_CLASSES}
        metrics = tm.MetricCollection(
            {
                'acc': tm.Accuracy(**config),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_metrics/')
        self.val_metrics = metrics.clone(prefix='val_metrics/')
        self.test_metrics = metrics.clone(prefix='test_metrics/')
        self.test_conf = tm.ConfusionMatrix(**config)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')

    def step(self, batch, batch_idx, stage='train'):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(
            y_pred,
            y,
            weight=torch.tensor([1.0, 1.0, 1.0, 1.05]).cuda(),
            # label_smoothing=0.01,
        )

        self.log(
            f'{stage}_loss',
            loss,
            prog_bar=True,
            on_step=(stage == 'train'),
            on_epoch=(stage != 'train'),
        )

        (metrics := getattr(self, f'{stage}_metrics'))(y_pred, y)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        if stage == 'test':
            self.test_conf.update(y_pred, y)

        return loss

    def on_test_end(self) -> None:
        fig_, ax_ = self.test_conf.plot()
        print('saving conf matrix...')
        plt.savefig(os.path.join(self.logger.log_dir, 'conf.png'))
        return super().on_test_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=0.1,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.5,
        )

        return [optimizer], [scheduler]


class RPSDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_transforms: list,
        transforms: list,
        batch_size: int = 32,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.transforms = transforms

    def setup(self, stage: str):
        if stage in (None, 'fit', 'predict'):
            self.train = tv.datasets.ImageFolder(
                self.root_dir / 'train',
                transform=self.train_transforms,
            )
            self.val = tv.datasets.ImageFolder(
                self.root_dir / 'val',
                transform=self.transforms,
            )

        if stage in (None, 'test', 'predict'):
            self.test = tv.datasets.ImageFolder(
                self.root_dir / 'test',
                transform=self.transforms,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKER,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.test,
            batch_size=self.batch_size,
        )


# means, stds = [0.8497, 0.8211, 0.8112], [0.2472, 0.2887, 0.3032]
means, stds = [0.6], [0.2]
FILL = 1.0  # 0.61
train_transforms = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
        tv.transforms.RandomApply(
            [
                tv.transforms.RandomAffine(
                    60,
                    translate=(0.0, 0.1),
                    scale=(0.75, 1.1),
                    fill=FILL,
                ),
            ],
            p=0.6,
        ),
        tv.transforms.ColorJitter(0.2),
        tv.transforms.Grayscale(1),
        tv.transforms.Normalize(means, stds),
        tv.transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            antialias=True,
        ),
    ]
)

transforms = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Grayscale(1),
        tv.transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True),
        tv.transforms.Normalize(means, stds),
    ]
)

datamodule = RPSDataModule(
    'data',
    batch_size=32,
    train_transforms=train_transforms,
    transforms=transforms,
)

# datamodule.setup('predict')
# import matplotlib.pyplot as plt

# sample_images, sample_labels = next(iter(datamodule.train_dataloader()))
# sample_images, sample_labels = (
#     sample_images.cpu().numpy(),
#     sample_labels.cpu().numpy(),
# )
# print(
#     f"images.shape: {sample_images.shape} - labels.shape: {sample_labels.shape}"
# )

# # plt.figure(figsize=(5,5))
# for i in range(32):
#     plt.subplot(4, 8, i + 1)
#     image = sample_images[i]
#     # print(f"images[{i}].shape: {image.shape} ")
#     image = image.transpose((1, 2, 0))
#     # print(f" - AP: images[{i}].shape: {image.shape}")
#     # plt.imshow(image.squeeze(), cmap='gray')
#     plt.text(0, 0, sample_labels[i])
#     plt.imshow(image * 255, cmap='gray', vmin=0, vmax=255)  # image.squeeze())
#     plt.axis('off')
# plt.show()
# plt.close()

# from tqdm import tqdm

# psum = torch.tensor([0.0, 0.0, 0.0])
# psum_sq = torch.tensor([0.0, 0.0, 0.0])

# dataloader = data_module.train_dataloader()
# for batch in tqdm(dataloader):
#     x, y = batch
#     psum += x.sum(axis=[0, 2, 3])
#     psum_sq += (x**2).sum(axis=[0, 2, 3])

# count = len(dataloader) * IMAGE_HEIGHT * IMAGE_WIDTH

# # mean and std
# total_mean = psum / count
# total_var = (psum_sq / count) - (total_mean**2)
# total_std = torch.sqrt(total_var)

# # output
# print('mean: ' + str(total_mean))
# print('std:  ' + str(total_std))

# from lightning.pytorch.loggers import TensorBoardLogger

# for i, (x, y) in enumerate(datamodule.val_dataloader(), 10000):
#     img = x.numpy().squeeze().transpose((1, 2, 0))
#     plt.imsave(f'data/val/{CLASSES[y]}/trans-{i}.png', img)

if __name__ == '__main__':
    # raise ValueError()
    torch.set_float32_matmul_precision('medium')

    model = RPSCNN()
    trainer = L.Trainer(
        default_root_dir='logs',
        # logger=TensorBoardLogger('logs'),
        callbacks=[
            (
                ckpt := callbacks.ModelCheckpoint(
                    filename='{epoch}',
                    every_n_epochs=1,
                    monitor=MONITOR,
                    mode='min',
                )
            ),
            callbacks.TQDMProgressBar(refresh_rate=1),
        ],
        max_epochs=NUM_EPOCHS,
    )

    print(f'Training for {NUM_EPOCHS} epochs.')
    t_start = time.time()
    trainer.fit(model, datamodule=datamodule)
    duration = time.time() - t_start
    print(f'Trained model in {duration:.2f}s')

    if not (match := re.search(r'(?<=epoch\=)(\d+)', ckpt.best_model_path)):
        raise RuntimeError('No valid checkpoint found.')
    best_epoch = int(match.group(0))

    print(
        f'Best : {MONITOR} {ckpt.best_model_score or 0.0:.4f} '
        f'(Epoch: {best_epoch})'
    )

    trainer.test(ckpt_path=ckpt.best_model_path, datamodule=datamodule)
