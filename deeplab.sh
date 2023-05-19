# create conda environment
conda create --name deeplab_pytorch python=3.7
conda activate deeplab_pytorch

#clone pytorch deeplab github repository
git clone https://github.com/CzJaewan/deeplabv3_pytorch-ade20k.git
cd deeplabv3_pytorch-ade20k
pip install -r requirements.txt

#correct row
sed -i '118 i \        num_mask = np.array(target)' datasets/ade20k.py


#install ADE20K dataset
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip -d ./datasets/data/ade20k/
rm ADEChallengeData2016.zip

#create odgt files for training process
mv ../makeODGT.py datasets/data/ade20k/
cd datasets/data/ade20k/
python makeODGT.py