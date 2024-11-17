import torch
from torch import nn
from tqdm import tqdm

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.fc = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.fc(x.reshape(x.shape[0], -1))
        x = nn.functional.sigmoid(x)
        return x


def mnist_images(path, cnt):
    with open(path, "rb") as f:
        train_images = torch.frombuffer(bytearray(f.read()[16:]), dtype=torch.uint8)
        return train_images.reshape(cnt, 1, 28, 28).to(torch.float32) / 255.0
def mnist_labels(path):
    with open(path, "rb") as f:
        train_labels = torch.frombuffer(bytearray(f.read()[8:]), dtype=torch.uint8)
        return nn.functional.one_hot(train_labels.to(torch.int64), num_classes=10).squeeze_().to(torch.float32)

train_images = mnist_images("./MNIST/raw/train-images-idx3-ubyte", 60_000)
test_images  = mnist_images("./MNIST/raw/t10k-images-idx3-ubyte", 10_000)
train_labels = mnist_labels("./MNIST/raw/train-labels-idx1-ubyte")
test_labels  = mnist_labels("./MNIST/raw/t10k-labels-idx1-ubyte")

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

@torch.no_grad()
def calc_acc():
    out = net(test_images)
    match = torch.argmax(out, axis=1) == torch.argmax(test_labels, axis=1)
    print(match.sum())


calc_acc()
for i in (pbar := tqdm(range(train_images.shape[0]))):
    optimizer.zero_grad()
    out = net(train_images[i].unsqueeze(1))
    loss = ((train_labels[i] - out) ** 2).mean()
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss={loss.item():.4f}")
    if i % 1000 == 0: calc_acc()

for k, v in net.state_dict().items():
    fn = f'conv_weights/{k}.bin'
    print("Saving to", fn)
    v.detach().numpy().tofile(fn)

print(net(train_images[0].unsqueeze(0)))
print(net(train_images[1].unsqueeze(0)))
print(net(train_images[2].unsqueeze(0)))
