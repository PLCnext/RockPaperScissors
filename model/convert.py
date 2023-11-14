import torch
from train import RPSCNN

# import torch.nn as nn
# import torchvision as tv

# from typing import Callable, Any

version = 72
epoch = 72
PATH = f'logs/lightning_logs/version_{version}/checkpoints/epoch={epoch}.ckpt'

model = RPSCNN.load_from_checkpoint(PATH)
model.eval()

# transforms = [
#     tv.transforms.ConvertImageDtype(dtype=torch.float32),
#     tv.transforms.Grayscale(num_output_channels=1),
#     tv.transforms.Resize((300, 300), antialias=False),
#     tv.transforms.Normalize(mean=[0.6], std=[0.2]),
# ]

# class Predictor(nn.Module):
#     def __init__(
#         self,
#         model: RPSCNN,
#         transforms: list[Callable[[Any], Any]],
#     ) -> None:
#         super().__init__()
#         self.model = model.eval()
#         self.transforms = nn.Sequential(*transforms)  # type: ignore

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             x = self.transforms(x)
#             return self.model(x)


# model = Predictor(model, transforms) # type: ignore

x = torch.rand(1, 1, 300, 300).cuda()
predictions = model(x)

torch.onnx.export(
    model,
    x,
    f'../src/static/model/cnn_v{version}.onnx',
    input_names=['input'],
    output_names=['scores'],
    opset_version=11,
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'scores': {0: 'batch'},
    },
)
