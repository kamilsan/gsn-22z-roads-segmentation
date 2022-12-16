import torch

import torchvision
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage, InterpolationMode

from project.lightning_module.segmentation_module import SegmentationModule

import hydra

from project.utils.labels import trainId2label
from project.utils.remap_labels import RemapCityscapesLabels

import os

from tqdm import tqdm

from omegaconf import DictConfig


@hydra.main(config_path='config', config_name='defaults', version_base='1.1')
def main(cfg: DictConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms_input = Compose([
        Resize(cfg.data.image_size),
        ToTensor(),
        Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std)
    ])

    relabeling_lut = [
        trainId2label[i].id if i in trainId2label else 0 for i in range(256)]
    original_size = (1024, 2048)

    to_target_transform = Compose([
        Resize(original_size, InterpolationMode.NEAREST),
        ToPILImage()
    ])

    relabeling_transform = RemapCityscapesLabels(relabeling_lut)

    dataset = torchvision.datasets.Cityscapes('./dataset', split='test', mode='fine',
                                              target_type='semantic', transform=transforms_input, target_transform=ToTensor())

    model = SegmentationModule.load_from_checkpoint('unet.ckpt').to(device)

    output_dir = 'test_set_predictions'
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for (x, _), image_path in tqdm(zip(dataset, dataset.images), total=len(dataset)):
            output = model(x.unsqueeze(dim=0).to(device))
            preds = torch.argmax(output, dim=1).unsqueeze(
                dim=1).to(torch.uint8)

            image_name = os.path.basename(image_path)
            output_image_path = os.path.join(output_dir, image_name)

            image = to_target_transform(preds.squeeze(dim=0))
            image_relabeled = relabeling_transform(image)
            image_relabeled.save(output_image_path)


if __name__ == '__main__':
    main()
