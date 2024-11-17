import torch
import numpy as np
torch.manual_seed(0)


def printv(name, t):
    t = t.detach().numpy()
    kwargs = {
        "separator": ", ",
        "formatter": {"float": lambda x: (" " if x >= 0 else "") + f"{x:.4f}"}
    }
    data = np.array2string(t, **kwargs).replace("["," ").replace("]","")
    sh = "*".join([str(i) for i in t.shape])
    print(f"{name}.data = [{sh}]f32{{")
    print(data)
    print("}")


print("==== mul ====")
a = torch.randn((3, 5))
b = torch.randn((5, 4))
printv("a", a)
printv("b", b)
printv("r", a@b)


print("==== mulT ====")
a = torch.randn((3, 5))
b = torch.randn((4, 5))
printv("a", a)
printv("b", b)
printv("r", a@b.T)


print("==== conv2d_simple ====")
img = torch.randn((1, 5, 5))
conv = torch.nn.Conv2d(1, 2, kernel_size=3)
printv("i", img)
printv("w", conv.weight)
printv("b", conv.bias)
printv("r", conv(img))


print("==== conv2d_big ====")
img = torch.randn((2, 5, 5))
conv = torch.nn.Conv2d(2, 3, kernel_size=3)
printv("i", img)
printv("w", conv.weight)
printv("b", conv.bias)
printv("r", conv(img))


print("==== conv2d_stride_simple ====")
img = torch.randn((2, 6, 6))
conv = torch.nn.Conv2d(2, 3, kernel_size=3, stride=2)
printv("i", img)
printv("w", conv.weight)
printv("b", conv.bias)
printv("r", conv(img))


print("==== conv2d_stride_big ====")
img = torch.randn((2, 7, 7))
conv = torch.nn.Conv2d(2, 3, kernel_size=4, stride=3)
printv("i", img)
printv("w", conv.weight)
printv("b", conv.bias)
printv("r", conv(img))
