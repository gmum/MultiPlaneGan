## MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation

This repo contains implementation of GAN version of ["MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation"](https://arxiv.org/pdf/2305.10579.pdf). It's built on top of [EG3D](https://github.com/NVlabs/eg3d).

### Installation
To install, run the following commands:
```
cd eg3d
conda env create -f environment.yml
conda activate eg3d
```

### Training and Datasets

Follow instructions from [EG3D](https://github.com/NVlabs/eg3d). 
The only differens is that we set the set the `gamma` parameter to 1.0.
