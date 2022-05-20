# PointNet Transfer Learning

Transfer learning with a [PointNet](https://arxiv.org/abs/1612.00593) is applied.

## Prerequisites

Before starting the training, you have to download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html) and extract it to an arbitrary directory. 
After that, create an environment variable *S3DIS_DIR* which points to the root directory of the dataset (e.g. *S3DIS_DIR = your/path/to/unzipped/files/AlignedVersion*). Create a folder *S3DIS_DIR/data* and copy all AREAs into it. For instance, *S3DIS_DIR/data/Area1* should be valid path.

Make sure that python >= 3.6.8 is available. 

pip install -r requirements.txt

## Quickstart

The next step is to prepare the dataset for the training. Apply the following commands

*python s3dis_prepare.py*

*python s3dis_prepare.py --mode blocks*

A directory *./Scenes/S3DIS* will be created where the point clouds and their corresponding labels are stored. The blocks which are used to train the PointNet are stored in *Blocks/S3DIS*. 
After that, the training can be started by calling

*python train.py*

which will train a semantic segmentation PointNet from scratch. Call

*python train.py -h*

in order to print the available command line options.

## Transfer Learning

The command line options that are relevant for the transfer learning are:

* load: If True, a model which is located at model_dir/model_file (see next two options) will be loaded.
* model_dir: Directory of the model that should be loaded.
* model_file: File name of the model that should be loaded.
* freeze: If True, the weights of the PointNet feature extractor will not be updated during the training.

### Example

Assume you already trained a model with dataset A which is located at model_dir/model_file with 

*python train.py --dataset A*

and you want to use the PointNet feature extractor of that model in order to train on dataset B. To do so, call, e.g.,

*python train.py --dataset B --load True --model_dir Path/To/Model --model_file Filename --freeze True*

which only changes the last layers of the model in the training. 

## DATASETS

* PCG: 12 Classes
* S3DIS: 14 Classes