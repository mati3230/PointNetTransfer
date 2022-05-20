# PointNet Transfer Learning

Transfer learning with a [PointNet](https://arxiv.org/abs/1612.00593) is applied.

## Prerequisites

Before starting the training, you have to download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html) and extract it to an arbitrary directory. 
After that, create an environment variable *S3DIS_DIR* which points to the root directory of the dataset (e.g. *S3DIS_DIR = your/path/to/unzipped/files/AlignedVersion*). Create a folder *S3DIS_DIR/data* and copy all AREAs into it. For instance, *S3DIS_DIR/data/Area1* should be valid path.

Make sure that python >= 3.6.8 is available. 

pip install -r requirements.txt

## Quickstart

The next step is to prepare the dataset for the training. Apply the script 

*python s3dis_prepare.py*. 

A directory *./Scenes/S3DIS* will be created where the blocks of the point clouds and their corresponding labels are stored.
After that, the training can be started by calling

*python train.py*

which will train a semantic segmentation PointNet from scratch. 

## Transfer Learning

TODO

## DATASETS

* PCG: 12 Classes
* S3DIS: 14 Classes