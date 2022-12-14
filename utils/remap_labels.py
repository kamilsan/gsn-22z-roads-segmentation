from PIL.Image import Image
from utils.labels import Label


class RemapCityscapesLabels:
    def __init__(self, lut_map: dict[int, Label]) -> None:
        self.lut = [
            lut_map[i].trainId if i in lut_map else 0 for i in range(256)]

    def __call__(self, img: Image) -> Image:
        return img.point(self.lut)
