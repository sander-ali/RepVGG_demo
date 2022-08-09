import torch
from utils import TwoBranches, RepVGG, get_fused_bn_to_conv_state_dict
from torch import nn
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(71)

two_branches = TwoBranches(8, 8)

x = torch.randn((1, 8, 7, 7))

two_branches(x).shape

conv1 = two_branches.conv1
conv2 = two_branches.conv2

conv_fused = nn.Conv2d(conv1.in_channels, conv1.out_channels, kernel_size=conv1.kernel_size)

conv_fused.weight = nn.Parameter(conv1.weight + conv2.weight)
conv_fused.bias =  nn.Parameter(conv1.bias + conv2.bias)

# check they give the same output
assert torch.allclose(two_branches(x), conv_fused(x), atol=1e-5)

two_branches.to("cuda")
conv_fused.to("cuda")

with torch.no_grad():
    x = torch.randn((4, 8, 7, 7), device=torch.device("cuda"))
    
    start = perf_counter()
    two_branches(x)
    print(f"conv1(x) + conv2(x) tooks {perf_counter() - start:.6f}s")
    
    start = perf_counter()
    conv_fused(x)
    print(f"conv_fused(x) tooks {perf_counter() - start:.6f}s")

conv_bn = nn.Sequential(
    nn.Conv2d(8, 8, kernel_size=3, bias=False),
    nn.BatchNorm2d(8)
)

torch.nn.init.uniform_(conv_bn[1].weight)
torch.nn.init.uniform_(conv_bn[1].bias)

with torch.no_grad():
    # be sure to switch to eval mode!!
    conv_bn = conv_bn.eval()
    conv_fused = nn.Conv2d(conv_bn[0].in_channels, 
                           conv_bn[0].out_channels, 
                           kernel_size=conv_bn[0].kernel_size)



    conv_fused.load_state_dict(get_fused_bn_to_conv_state_dict(conv_bn[0], conv_bn[1]))

    x = torch.randn((1, 8, 7, 7))
    
    assert torch.allclose(conv_bn(x), conv_fused(x), atol=1e-5)

with torch.no_grad():
    x = torch.randn((1,2,3,3))
    identity_conv = nn.Conv2d(2,2,kernel_size=3, padding=1, bias=False)
    identity_conv.weight.zero_()
    print(identity_conv.weight.shape)

    in_channels = identity_conv.in_channels
    for i in range(in_channels):
        identity_conv.weight[i, i % in_channels, 1, 1] = 1

    print(identity_conv.weight)
    
    out = identity_conv(x)
    assert torch.allclose(x, out)

def init(repvgg):
    for module in repvgg.modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    return repvgg

def benchmark(batches_sizes, device="cpu"):
    records = []
    # test the models
    with torch.no_grad():
        x = torch.randn((1, 3, 112, 112))
        repvgg = init(RepVGG([4, 8, 16], [1, 1, 1])).eval()
        out = repvgg(x)
        out_fast = repvgg.switch_to_fast()(x)
        assert torch.allclose(out, out_fast, atol=1e-5)

    print(f"{device=}")
    for batch_size in batches_sizes:
        x = torch.randn((batch_size, 3, 224, 224), device=torch.device(device))
        torch.cuda.reset_peak_memory_stats
        with torch.no_grad():
            repvgg = (
                RepVGG([2, 4, 8, 16, 32, 64, 128, 256, 512], [2, 2, 2, 2, 2, 2, 2, 2, 2])
                .eval()
                .to(torch.device(device))
            )
            start = perf_counter()
            for _ in range(32):
                repvgg(x)

            records.append(
                {
                    "Type": "Conventional",
                    "VRAM (B)": torch.cuda.max_memory_allocated(),
                    "Time (s)": perf_counter() - start,
                    "batch size": batch_size,
                    "device": device,
                }
            )
            print(
                f"Memory without RepVGG {torch.cuda.max_memory_allocated()}"
            )
            print(f"Without RepVGG {perf_counter() - start:.2f}s")
            torch.cuda.reset_peak_memory_stats
            repvgg.switch_to_fast().to(torch.device(device))
            start = perf_counter()
            for _ in range(32):
                repvgg(x)

            records.append(
                {
                    "Type": "Fast RepVGG",
                    "VRAM (B)": torch.cuda.max_memory_allocated(),
                    "Time (s)": perf_counter() - start,
                    "batch size": batch_size,
                    "device": device,
                }
            )
            print(f"With RepVGG {perf_counter() - start:.2f}s")
            print(f"Memory with RepVGG {torch.cuda.max_memory_allocated()}")

    return pd.DataFrame.from_records(records)


# df = benchmark([1, 2, 4, 8, 16, 32, 64, 128], "cuda")


if __name__ == "__main__":

    plt.style.use(['science','no-latex'])

    batches_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    df = benchmark(batches_sizes, "cuda")
    print(df)

    fig = plt.figure()

    default_time = df[df.loc[:, "Type"] == "Conventional"].loc[:, "Time (s)"]
    fast_time = df[df.loc[:, "Type"] == "Fast RepVGG"].loc[:, "Time (s)"]

    plt.plot(batches_sizes, default_time.values, label="Conventional")
    plt.plot(batches_sizes, fast_time.values, label="RepVGG")

    plt.xlabel("Varying Batch Size")
    plt.ylabel("Execution Time in seconds")
    plt.legend()
    plt.savefig("time.png", dpi=800)

    fig = plt.figure()

    default_time = df[df.loc[:, "Type"] == "Conventional"].loc[:, "VRAM (B)"]
    fast_time = df[df.loc[:, "Type"] == "Fast RepVGG"].loc[:, "VRAM (B)"]

    plt.plot(batches_sizes, default_time.values, label="Conventional")
    plt.plot(batches_sizes, fast_time.values, label="RepVGG")

    plt.xlabel("Varying Batch Size")
    plt.ylabel("VRAM (B)")
    plt.legend()
    plt.savefig("vram.png", dpi=800)