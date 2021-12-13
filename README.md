# CyCTR-PyTorch
This is a PyTorch re-implementation of NeurIPS 2021 paper "[Few-Shot Segmentation via Cycle-Consistent Transformer](https://proceedings.neurips.cc/paper/2021/file/b8b12f949378552c21f28deff8ba8eb6-Paper.pdf)".

# Usage

### Requirements
```
Python==3.8
GCC==5.4
torch==1.6.0
torchvision==0.7.0
cython
tensorboardX
tqdm
PyYaml
opencv-python
pycocotools
```

#### Build Dependencies
```
cd model/ops/
bash make.sh
cd ../../
```

### Data Preparation

+ PASCAL-5^i: Please refer to [PFENet](https://github.com/dvlab-research/PFENet) to prepare the PASCAL dataset for few-shot segmentation. 

+ COCO-20^i: Please download COCO2017 dataset from [here](https://cocodataset.org/#download). Put or link the dataset to ```YOUR_PROJ_PATH/data/coco```. And make the directory like this:

```
${YOUR_PROJ_PATH}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   `-- instances_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```

Then, run  
```
python prepare_coco_data.py
```
to prepare COCO-20^i data.

### Train
Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.

Then, run this command: 
```
    sh train.sh {*dataset*} {*model_config*}
```
For example, 
```
    sh train.sh pascal split0_resnet50
```

### Test Only
+ Download checkpoints from [here](https://drive.google.com/drive/folders/1P3Qo7Zz_257z9gnVb7wroV7acaFYinkw?usp=sharing)
+ Modify `config` file (specify checkpoint path)
+ Run the following command: 
```
    sh test.sh {*dataset*} {*model_config*}
```

For example, 
```
    sh test.sh pascal split0_resnet50
```

Results on 1-shot Pascal-5^i
| Model              | Split-0 | Split-1 | Split-2 | Split-3 |  Mean | 
|--------------------|---------|---------|---------|---------|-------|
| CyCTR_resnet50     | 67.8    |  72.7   |  58.0   |  57.9   |  64.1 | 

# Acknowledgement

This project is built upon [PFENet](https://github.com/dvlab-research/PFENet) and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), thanks for their great works!

# Citation

If you find our codes or models useful, please consider to give us a star or cite with:
```
@article{zhang2021few,
  title={Few-Shot Segmentation via Cycle-Consistent Transformer},
  author={Zhang, Gengwei and Kang, Guoliang and Wei, Yunchao and Yang, Yi},
  journal={arXiv preprint arXiv:2106.02320},
  year={2021}
}
```
