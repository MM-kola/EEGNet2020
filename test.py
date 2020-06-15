import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import load_image
from flashtorch.saliency import Backprop
from flashtorch.utils import apply_transforms
image = load_image("G:\EEGNet/test/test.jpg")

plt.imshow(image)

net = models.vgg16(pretrained=1)

backprop = Backprop(net)

input_ = apply_transforms(image)

target_class = 24

backprop.visualize(input_, target_class, guided=True)