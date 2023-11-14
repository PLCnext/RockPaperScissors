# Folder Structure

The folder structure is as follows:

```{bash}
.
├── convert.py # convert model to ONNX
├── data       # training data
├── logs       # training logs
├── onnx       # converted models (old)
└── train.py   # train a model
```

# Prerequisites

Install the required packages with pip:

```{bash}
pip install -r requirements.txt
```

Then install `torch==2.0.1` and `torchvision==0.15.2` or the latest version from [PyTorch](https://pytorch.org/).

# Train a model

You can train a model with:

```{bash}
python model.py
```

Then the model needs to converted to [ONNX](https://onnx.ai/).
Adjust the `version` to match the folder `logs/version_{version}` and `epoch` to match the name of the checkpoint `checkpoints/epoch={epoch}.ckpt`.
Then run the following command to convert the model:

```{bash}
python convert.py
```

The model will be saved under `src/static/model/cnn_{version}.onnx`.
Now you can change the environment variable `MODEL_VERSION` to match your `version`.
