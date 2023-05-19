# Creating-dataset-via-SDM

[Paper](https://comserv.cs.ut.ee/ati_thesis/datasheet.php?id=77573)


## Generating images 

To set up environment to start [semantic diffusion model](https://github.com/WeilunWang/semantic-diffusion-model) 

```bash
# From semantic-diffusion-model directory
bash scripts/setup.sh
```

Before starting to generate images we have to download [ADE20K trained checkpoint](https://drive.google.com/file/d/1O8Avsvfc8rP9LIt5tkJxowMTpi1nYiik/view) and also ADE20K dataset by running script 

```bash
# From semantic-diffusion-model/data/ 
bash download_ade20k.sh
```

All parameters are written inside of python file image_sample, so to start generating images simply run:

```bash
# From semantic-diffusion-model 
python image_sample.py
```

With default parameters images are generated based on ADE20K training set.


Images generated with semantic diffusion model are available on [this OneDrive folder](https://1drv.ms/f/s!Aum-f-ncjGIvlEas-6in7RPqJmHN?e=B2EOh0).

## DeepLabv3+ segmentation model

For setting up the [DeepLabv3](https://github.com/CzJaewan/deeplabv3_pytorch-ade20k) repository run:

```bash
# From base directory (Creating-dataset-via-sdm)
bash deeplab.sh
```

For training:
```bash
# From deeplabv3_pytorch-ade20k
bash deeplab_train.sh
```

In the same script there is example commented out how to predict segments on wanted images.
After adding the generated images, we train a second model to compare the results.
