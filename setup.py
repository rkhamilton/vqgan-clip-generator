from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='vqgan-clip-generator',
    version='1.0.0',
    description='Implements VQGAN+CLIP and support functions for image and video generation based on text and image prompts',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rkhamilton/vqgan-clip-generator',
    author='Ryan Hamilton',
    author_email='ryan.hamilton@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Environment :: GPU :: NVIDIA CUDA',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='VQGAN, VQGAN+CLIP, deep dream, neural network, pytorch',  # Optional
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.6, <4',
    install_requires=[
        'ftfy',
        'regex',
        'tqdm',
        'pytorch-lightning',
        'kornia',
        'imageio',
        'omegaconf',
        'taming-transformers',
        'torch_optimizer'],
    extras_require={
        'dev' : ['pytest','setuptools'],
        'test': ['pytest'],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/rkhamilton/vqgan-clip-generator/issues',
        'Source': 'https://github.com/rkhamilton/vqgan-clip-generator',
    },
)