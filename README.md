
<p align="center">
  <h1 align="center">LM2CNet: Enhancing Monocular 3D Visual Grounding with Language Guided Multi-Modality Coupling Network</h1>
  <p align="center">


   <br />
    <strong>Meng Li</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    ·
    <a href="https://cv-shuchanglyu.github.io/EnHome.html"><strong>Shuchang Lyu</strong></a>
    ·
    <strong>Jun Jiang</strong></a>    
    ·
    <strong>Longhao Zou</strong></a>
    ·
    <a href="https://sites.google.com/view/guangliangcheng"><strong>Guangliang Cheng</strong></a>
    ·
    <strong>Chenguang Yang</strong></a>
    <br />
<p align="center">

    
  </p>





## Highlight!!!!
This repo is the implementation of "LM2CNet: Enhancing Monocular 3D Visual Grounding with Language Guided Multi-Modality Coupling Network". We refer to [Mono3DVG](https://github.com/ZhanYang-nwpu/Mono3DVG), [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), [nuScenes](https://github.com/nutonomy/nuscenes-devkit), [DriveLM](https://github.com/OpenDriveLab/DriveLM). Many thanks to these excellent repos.

## TODO
- [x] Release test_dataset
- [x] Release checkpoints
- [ ] Release all_dataset

## Installation
* Ubantu==18.04
* Python==3.10 
* Torch==1.12.1, Torchvision==0.12.0
* CUDA==11.3
* checkpoint==[LM2CNet](https://drive.google.com/file/d/1auMd9sOpYcAaIelJVKPOKvBYKic7yy4w/view?usp=drive_link)
* Test_Dataset==[Test_Dataset](https://drive.google.com/file/d/1a-U9jg_xd2BDMQk8Fk8hJqYBOYS9iv01/view?usp=drive_link)
* Bert_Pretrain==[Bert](https://drive.google.com/file/d/1ee-XVDnqTNj3tBqgc1S2WLFEMY2dv2iU/view?usp=drive_link)


**Please add the Bert_Pretrain into ./roberta-base folder**


**Please add the checkpoint into ./config folder**
```
conda create -n LM2CNet python=3.10
conda activate LM2CNet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

```
cd /LM2CNet
pip install -r requirments.txt
```

```
cd /LM2CNet/lib/models/LM2CNet/ops
bash make.sh
```
## Test on Mono3DRefer_nuScenes
**please set the Test_Dataset path**


**please set the checkpoint path**
```
cd /LM2CNet
python test.py
```
## Demo Video

[Demo](https://github.com/user-attachments/assets/9e6e3e33-5ebb-4dd2-9f3f-83f46848e5e6)

## Dataset Construction
**If you want to use chatgpt to automatically generate language description, we have provided a demo tool in folder /LM2CNet/datasets_construction for your reference.**


**If you want to use DriveLM to indentify the nuScens object, we have provided a demo tool in folder /LM2CNet/datasets_construction for your reference.**

## Others
**If you have any problems, please contact me limenglm@buaa.edu.cn**
