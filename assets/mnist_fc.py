import torch
from torch import nn
from tqdm import tqdm

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


with open("./MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    train_images = torch.frombuffer(bytearray(f.read()[16:]), dtype=torch.uint8)
    train_images = train_images.reshape(60_000, 28*28).to(torch.float32) / 255.0
with open("./MNIST/raw/train-labels-idx1-ubyte", "rb") as f:
    train_labels = torch.frombuffer(bytearray(f.read()[8:]), dtype=torch.uint8)
    train_labels = nn.functional.one_hot(train_labels.to(torch.int64), num_classes=10).squeeze_().to(torch.float32)

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for i in (pbar := tqdm(range(train_images.shape[0]))):
    optimizer.zero_grad()
    out = net(train_images[i])
    loss = ((train_labels[i] - out) ** 2).mean()
    loss.backward()
    optimizer.step()

net.fc1.weight.detach().numpy().tofile('fc_weights/fc1.weight.bin')
net.fc1.bias  .detach().numpy().tofile('fc_weights/fc1.bias.bin')
net.fc2.weight.detach().numpy().tofile('fc_weights/fc2.weight.bin')
net.fc2.bias  .detach().numpy().tofile('fc_weights/fc2.bias.bin')
