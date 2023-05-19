#script to set up conda environment and install required packages
conda create --name sdm python=3.7

conda activate sdm

pip install blobfile=2.0.2 \
mpi4py=3.1.4 \
torch=1.13.1 \
torchvision=0.14.1 \
tqdm=4.65.0 \
requests=2.29.0
