import pytest
from os import path, remove
import vqgan_clip.download


def test_imagenet_f16_16384():
    vqgan_clip.download.imagenet_f16_16384_dummy()
    file1 = 'checkpoints/vqgan_imagenet_f16_16384.yaml'
    file2 = 'checkpoints/vqgan_imagenet_f16_16384.ckpt'
    assert path.exists(file1)
    assert path.exists(file2)
    remove(file1)
    remove(file2)