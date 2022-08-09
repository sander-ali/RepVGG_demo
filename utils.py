import torch
import warnings
import collections
from itertools import repeat
from torch import nn, Tensor
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")
    
class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )

class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )
        
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.weight = nn.Sequential(
            Conv2dNormActivation(
                in_channels, out_channels, kernel_size=3, stride=stride
            ),
            Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, activation_layer=None
            ),
        )
        self.shortcut = (
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                activation_layer=None,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)  # <- 2x memory
        x = self.weight(x)
        x += res
        x = self.act(x)  # <- 1x memory
        return x

class TwoBranches(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1 + x2

def get_fused_bn_to_conv_state_dict(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> Dict[str, Tensor]:
    # in the paper, weights is gamma and bias is beta
    bn_mean, bn_var, bn_gamma, bn_beta = (
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
    )
    # we need the std!
    bn_std = (bn_var + bn.eps).sqrt()
    # eq (3)
    conv_weight = nn.Parameter((bn_gamma / bn_std).reshape(-1, 1, 1, 1) * conv.weight)
    # still eq (3)
    conv_bias = nn.Parameter(bn_beta - bn_mean * bn_gamma / bn_std)
    return {"weight": conv_weight, "bias": conv_bias}

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            activation_layer=None,
            # the original model may also have groups > 1
        )

        self.shortcut = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            activation_layer=None,
        )

        self.identity = (
            nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x  # <- 2x memory
        x = self.block(x)
        x += self.shortcut(res)
        if self.identity:
            x += self.identity(res)
        x = self.relu(x)  # <- 1x memory
        return x

def get_fused_conv_state_dict_from_block(block: RepVGGBlock) -> Dict[str, Tensor]:
    fused_block_conv_state_dict = get_fused_bn_to_conv_state_dict(
        block.block[0], block.block[1]
    )

    if block.shortcut:
        # fuse the 1x1 shortcut
        conv_1x1_state_dict = get_fused_bn_to_conv_state_dict(
            block.shortcut[0], block.shortcut[1]
        )
        # we pad the 1x1 to a 3x3
        conv_1x1_state_dict["weight"] = torch.nn.functional.pad(
            conv_1x1_state_dict["weight"], [1, 1, 1, 1]
        )
        fused_block_conv_state_dict["weight"] += conv_1x1_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += conv_1x1_state_dict["bias"]
    if block.identity:
        # create our identity 3x3 conv kernel
        identify_conv = nn.Conv2d(
            block.block[0].in_channels,
            block.block[0].in_channels,
            kernel_size=3,
            bias=True,
            padding=1,
        ).to(block.block[0].weight.device)
        # set them to zero!
        identify_conv.weight.zero_()
        # set the middle element to zero for the right channel
        in_channels = identify_conv.in_channels
        for i in range(identify_conv.in_channels):
            identify_conv.weight[i, i % in_channels, 1, 1] = 1
        # fuse the 3x3 identity
        identity_state_dict = get_fused_bn_to_conv_state_dict(
            identify_conv, block.identity
        )
        fused_block_conv_state_dict["weight"] += identity_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += identity_state_dict["bias"]

    fused_conv_state_dict = {
        k: nn.Parameter(v) for k, v in fused_block_conv_state_dict.items()
    }

    return fused_conv_state_dict

class RepVGGFastBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
            activation_layer=None,
            # the original model may also have groups > 1
        )

        self.shortcut = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            activation_layer=None,
        )

        self.identity = (
            nn.BatchNorm2d(out_channels) if in_channels == out_channels else None
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x  # <- 2x memory
        x = self.block(x)
        x += self.shortcut(res)
        if self.identity:
            x += self.identity(res)
        x = self.relu(x)  # <- 1x memory
        return x

    def to_fast(self) -> RepVGGFastBlock:
        fused_conv_state_dict = get_fused_conv_state_dict_from_block(self)
        fast_block = RepVGGFastBlock(
            self.block[0].in_channels,
            self.block[0].out_channels,
            stride=self.block[0].stride,
        )

        fast_block.conv.load_state_dict(fused_conv_state_dict)

        return fast_block

class RepVGGStage(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
    ):
        super().__init__(
            RepVGGBlock(in_channels, out_channels, stride=2),
            *[RepVGGBlock(out_channels, out_channels) for _ in range(depth - 1)],
        )

class RepVGG(nn.Sequential):
    def __init__(self, widths: List[int], depths: List[int], in_channels: int = 3):
        super().__init__()
        in_out_channels = zip(widths, widths[1:])

        self.stages = nn.Sequential(
            RepVGGStage(in_channels, widths[0], depth=1),
            *[
                RepVGGStage(in_channels, out_channels, depth)
                for (in_channels, out_channels), depth in zip(in_out_channels, depths)
            ],
        )

        # omit classification head for simplicity

    def switch_to_fast(self):
        for stage in self.stages:
            for i, block in enumerate(stage):
                stage[i] = block.to_fast()
        return self

