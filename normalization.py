import torch
from tqdm import tqdm

import torchvision
import torchvision.transforms


def main():

    transforms_input = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    num_channels = 3
    dataset = torchvision.datasets.Cityscapes('./dataset', split='train', mode='fine',
                                              target_type='semantic', transform=transforms_input)

    sum_color = torch.zeros((num_channels,))
    sum_squared_color = torch.zeros((num_channels,))
    num_images = 0

    print(len(dataset))

    for im, _ in tqdm(dataset):
        im_pixels = im.view((num_channels, -1))
        sum_color += torch.sum(im_pixels, dim=1) / torch.numel(im_pixels)
        sum_squared_color += torch.sum(im_pixels *
                                       im_pixels, dim=1) / torch.numel(im_pixels)
        num_images += 1

    mean = sum_color / num_images
    avg_squared_color = sum_squared_color / num_images
    std = torch.sqrt(avg_squared_color - mean * mean)

    print(f'Mean: {mean}')
    print(f'Std: {std}')


if __name__ == '__main__':
    main()
