import torch
from torchvision import transforms

# Image to point cloud by threshold.
def im2pc(image):
    x_len, y_len = image[0].size()
    max = image.max()
    pc = []
    for x in range(x_len):
        for y in range(y_len):
            if image[0, x, y] > 0.95 * max:
                pc.append([x, y])
    point_cloud = torch.Tensor(pc)
    point_cloud[:, 0] = (point_cloud[:, 0] / x_len)
    point_cloud[:, 1] = (point_cloud[:, 1] / y_len)
    return point_cloud.unsqueeze(0)


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)
