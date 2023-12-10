# MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation

This repo contains implementation of GAN version of ["MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation"](https://arxiv.org/pdf/2305.10579.pdf). It's built on top of [EG3D](https://github.com/NVlabs/eg3d).

## Installation
To install, run the following commands:
```
cd multiplanegan
conda env create -f environment.yml
conda activate multiplanegan
```

## Training and Datasets

### Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix. Custom datasets can be created from a folder containing images; see `python dataset_tool.py --help` for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

**FFHQ**: Download and process the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) using the following commands.

1. Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) submodule is properly initialized
```.bash
git submodule update --init --recursive
```

2. Run the following commands
```.bash
cd dataset_preprocessing/ffhq
python runme.py
```

Optional: preprocessing in-the-wild portrait images. 
In case you want to crop in-the-wild face images and extract poses using [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) in a way that align with the FFHQ data above and the checkpoint, run the following commands 
```.bash
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
```

**ShapeNet Cars**: Download and process renderings of the cars category of [ShapeNet](https://shapenet.org/) using the following commands.
NOTE: the following commands download renderings of the ShapeNet cars from the [Scene Representation Networks repository](https://www.vincentsitzmann.com/srns/).

```.bash
cd dataset_preprocessing/shapenet
python runme.py
```

### Training

You can train new networks using `train.py`. For example:

```.bash
# Train with Shapenet from scratch, using 4 GPUs.
# To train for more than 3M images, increase --kimg or use --resume on the next run.
python train.py --outdir=~/training-runs --cfg=shapenet --data=~/datasets/cars_train.zip \
  --gpus=4 --mbstd-group=8 --batch=32 --gamma=1.0 --kimg=3000
```

Please see the [Training Guide](./docs/training_guide.md) for a guide to setting up a training run on your own data.

The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-ffhq-ffhq512-gpus8-batch32-gamma1`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.
