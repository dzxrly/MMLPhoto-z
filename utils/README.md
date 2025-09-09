# `PyTorch Lightning`训练工具包

## Requirements

- numpy
- pynvml
- lightning
- torch
- torchinfo
- rich

## Install Dependences

假设已经完成了`PyTorch`和`lightning`安装！

```bash
pip install pynvml torchinfo
```

---

## `convert_str_params_list`

将`Wandb`传入的`str`根据指定分隔符转译为指定数据类型的`list`。

### Params

- `str_params`: `str`，`Wandb`输入的字符串。
- `params_type`: `str`，转译后的数据类型。
- `split_marker`: `str`，default: `#`，分隔符。

### Return

- `list<eval('params_type')>`，使用指定数据类型的数组。

---

## `set_random_seed`

为`PyTorch`和`Pytorch_lightning`设置固定随机种子。

### Params

- `random_seed`: `float / int`，指定的随机种子。

### Return

- `null`

---

## `predict_model_memory_usage`

使用`torchinfo`的`summary`函数预估在`Float32`格式下训练模型所需的显存占用（GB）。

### Params

- `model`: `torch.nn.Module`，指定的模型。
- `input_shape`: `list[...]`，与`summary`输入格式一致的模型输入数据形状。

### Return

- `float`，显存占用（GB）。

---

## `auto_find_memory_free_card`

**仅适用于单卡训练**，根据模型的显存占用情况自动选择剩余显存最大的显卡，如果显存均不足，则根据设定的等待时间（秒）自动等待直到超时退出或有满足要求的显卡激活训练过程。

### Params

- `card_list`: `list[int, ...]`，指定使用的显卡编号列表。
- `model_memory_usage`: `float`，`Float32`下模型的显存占用（GB）。
- `idle`: `bool`，default: `false`，是否启动等待模式；若不启用，则未找到满足显存要求的显卡时直接报错退出，若启用，则按照
  `idle_max_seconds`的设置等待空余显卡直到超时。
- `idle_max_seconds`: `int`，超时时长，仅在启用`idle`时生效。

### Return

- `int`，可供使用的显卡编号。

---

> Made By Egg Targaryen
>
> MIT License