import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def minst_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
  else:
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
  return transform_list

def cifar_transform(is_training=True):
  # Data No Normalzation(std and mean)
  if is_training:
    transform_list = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.Pad(4, padding_mode='reflect'),
                                         transforms.RandomCrop(32, padding=0),
                                         transforms.ToTensor()])
  else:
    transform_list = transforms.Compose([transforms.ToTensor()])

  return transform_list

def imgnet_transform(is_training=True):
  # Data No Normalzation(std and mean)
  if is_training:
    transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor()])
  else:
    transform_list = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
  return transform_list

# def cifar_transform(is_training=True):
#   # Data
#   if is_training:
#     transform_list = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                          transforms.Pad(4, padding_mode='reflect'),
#                                          transforms.RandomCrop(32, padding=0),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                               (0.2023, 0.1994, 0.2010))])
#   else:
#     transform_list = transforms.Compose([transforms.ToTensor(),
#                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                               (0.2023, 0.1994, 0.2010))])

#   return transform_list


# def imgnet_transform(is_training=True):
#   if is_training:
#     transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
#                                          transforms.RandomHorizontalFlip(),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                               std=[0.229, 0.224, 0.225])])
#   else:
#     transform_list = transforms.Compose([transforms.Resize(256),
#                                          transforms.CenterCrop(224),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                               std=[0.229, 0.224, 0.225])])
#   return transform_list

#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(object):
    """ 
    color jitter 
    source: https://github.com/liuzechun/MetaPruning
    """
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'