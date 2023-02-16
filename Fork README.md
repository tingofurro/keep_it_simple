# Apex Amp to Torch Amp

Since Apex Amp is deprecated and cumbersome to install, I switched the Apex Amp code to Torch Amp code using the
official [PyTorch documentation](https://github.com/pytorch/pytorch/issues/52279).

> Important notices
> PyTorch Amp does not use Apex Amp 02 but rather
> an [improved](https://github.com/NVIDIA/apex/issues/818#issuecomment-639012282) of
>
the [01 Apex Amp version](https://discuss.pytorch.org/t/how-can-i-use-o2-optimization-with-torch-cuda-amp-like-apex/156449).

# Modifications

1. Changed import of `apex.amp` in `train_keep_it_simple.py` to `torch.cuda.amp`.
2. Changed

```python
if self.use_apex:
    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
```

to

```python
if self.scaler is not None:
    self.scaler.scale(loss).backward()
    self.scaler.step(optimizer=self.optimizer)
    self.scaler.update()
```

using the `self.scaler = GradScaler()` new initialized in the `__ini__()` as
per [documentation example](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training).

3. Removed this
   since [`torch.amp` does not require this kind of initialization](https://discuss.pytorch.org/t/torch-cuda-amp-equivalent-of-apex-amp-initialize/132598/5).

```python
if use_torch_amp:
    simplifier.model, optimizer = amp.initialize(
        simplifier.model, optimizer, opt_level="O2"
    )  # O1 is really not good, according to experiments on 10/13/2020
```

# Deprecated Fix

## `add_`

Deprecated warning of the `add_` method in `utils_optim.py`:

```shell
optim.py:201: UserWarning: This overload of add_ is deprecated: add_(Number alpha, Tensor other)
Consider using one of the following signatures instead: add_(Tensor other, *, Number alpha) 
(Triggered internally at ../torch/csrc/utils/python_arg_parser.cpp:1420.) exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
```

Fixed by the following `exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)` for all `add_` call.

See [here](https://discuss.pytorch.org/t/userwarning-this-overload-of-add-addcmul-addcdiv-is-deprecated-errors-while-implementing-sharedadam/99802)
for more details.

## `AdamW`

Deprecated warning of the `transformers.adamW` in `utils_optim.py`:

```shell
FutureWarning: This implementation of AdamW is deprecated and will be removed in a 
future version. Use the PyTorch implementation torch.optim.AdamW instead, or set 
$(no_deprecation_warning=True) to disable this warning
```

Fixed by removing `transformer.AdamW` by recommended `torch.optim.adamw`.