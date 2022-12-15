from typing import Dict
from PIL.Image import Image
from project.utils.labels import Label


class RemapCityscapesLabels:
    def __init__(self, lut_map: Dict[int, Label]) -> None:
        self.lut = [
            lut_map[i].trainId if i in lut_map else 0 for i in range(256)]

    def __call__(self, img: Image) -> Image:
        return img.point(self.lut)
