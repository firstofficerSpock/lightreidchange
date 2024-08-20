#lightreidchange
#dataset
[Market1501](https://drive.google.com/file/d/1M8m1SYjx15Yi12-XJ-TV6nVJ_ID1dNN5/view?usp=sharing)

[DukeMTMC](https://drive.google.com/file/d/11FxmKe6SZ55DSeKigEtkb-xQwuq6hOkE/view?usp=sharing)

[sysu128x64](https://drive.google.com/file/d/15itcysqi0NPhXrhdeuJHmaC8LvsTIYrt/view?usp=sharing)

#how to use
cd lightreidchange/light-reid
conda create -n lightreid python=3.7

conda activate lightreid

pip install -r requirements

cd /root/light-reid/examples/bagtricks_buildwithconfigs

pip install timm==0.5.4

python train.py

#datasets path 
/lightreidchange/light-reid/lightreid/data/datasets

#change datasetpaths.yaml to your path

# change configs path and config file in train.py to meet your requirement

# config file in
/lightreidchange/light-reid/examples/bagtricks_buildwithconfigs/configs
