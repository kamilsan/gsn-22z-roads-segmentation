import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms

from models.unet import UNet


def main():

    transforms_input = torchvision.transforms.Compose([
        torchvision.transforms.Resize((572, 572)),
        torchvision.transforms.ToTensor()
    ])

    transforms_target = torchvision.transforms.Compose([
        torchvision.transforms.Resize((388, 388)),
        torchvision.transforms.PILToTensor()
    ])

    dataset = torchvision.datasets.Cityscapes('./dataset', split='train', mode='fine',
                                              target_type='semantic', transform=transforms_input, target_transform=transforms_target)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet(3, 30)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    first_batch = [next(iter(loader))]

    epochs = 1000

    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} out of {epochs}')

        for x, target in first_batch:
            y = model(x)

            # unsqueeze needed to make a batch, because squeeze removed it
            loss = loss_function(y, target.long().squeeze().unsqueeze(dim=0))
            print(f'Loss: {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()
