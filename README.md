This repo contains the official implementation of the ACM MM 2024 paper

# MFRGN: Multi-scale Feature Representation Generalization Network for Ground-to-Aerial Geo-localization

<center>Yuntao Wang, Jinpu Zhang, Ruonan Wei, Wenbo Gao, Yuehuan Wang*</center>

paper([MM'24](https://dl.acm.org/doi/10.1145/3664647.3681431))

This code is based on the [Sample4Geo](https://github.com/Skyy93/Sample4Geo) framework. 

**Details of the datasets, training and inference can be found in Sample4Geo.**


## News

* **12 Dec 2024** We now provide supplementary results on University-1652.

| Methods    | Drone2Sat<br />R@1 / AP | Sat2Drone<br />R@1 / AP |
| :--------- | :---------------------: | :---------------------: |
| Sample4Geo |      92.65 / 93.81      |      95.65 / 91.39      |
| Ours       |      94.33 / 95.24      |      96.15 / 93.94      |


### Dataset Preparation

To accelerate training/test time, you can run `data_preparation.py`， which  implements  image transformation (from '.jpg'/'.png' to '.pt') and cropping (similar to  [SAFA](https://github.com/shiyujiao/cross_view_localization_SAFA)). 

When you process images from '.jpg'/'.png' to '.pt', you should set `ext='pt'` in `sample4geo/dataset/*.py`

Also, if you are experiencing network errors about the backbone, you may need to download the backbone weights offline and put them into the `pretrained` folder.

### Results

We provide our pretrained results: [MFRGN-pretained.zip](https://pan.baidu.com/s/1LMYpQVHeV99u4u28jtlrjw) [BaiduYun, Password: 1234], which contains pretrained weight files or files necessary to train certain network configurations (e.g. distance_dict, convnext backbone weights).





