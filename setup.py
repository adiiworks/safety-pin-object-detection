
from setuptools import setup, find_packages

setup(
    name='green_box_detector',
    version='1.0.0',
    description='Faster R-CNN Object Detection and Counting for green_box objects (No YOLO)',
    author='Aditya',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python',
        'pycocotools',
        'torchmetrics'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)

