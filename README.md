# IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment
This is the source code for [IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment](https://www.frontiersin.org/articles/10.3389/fnins.2021.739138/full).![IE-IQA Framework](https://github.com/esnthere/IE-IQA/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
timm: 0.3.2  
CUDA: 10.2  

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The model pre-trained on KonIQ-10k is '**IEIQA_koniq.pt**'. The dataset are randomly splitted several times during training, and the released model is obtained from the first split. Please run '**test_example_of_koniq.py**' to make both cross-dataset and intra-dataset.  The model file '**my_efficientnet.py**' and '**my_efficientnet_ie_feature.py**' is modified from open accessed source code of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch). 
   
   
## For train:  
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   Please run '**train_example_of_koniq.py**' to train the model. The training example can be found in '**run test_example_of_koniq.ipynb**'. The model is based on the ImageNet pre-traind weight of '**efficientnet-b0.pth**', which is obtained from [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch). 
   
## Simplified version without feature selection:
   A simplified version without feature selection is also provided in the folder 'simplified version', which achieves similar performance. 

## If you like this work, please cite:

{   
      AUTHOR={Song, Tianshu and Li, Leida and Zhu, Hancheng and Qian, Jiansheng},  
      TITLE={IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment},     
      JOURNAL={Frontiers in Neuroscience},     
      VOLUME={15},         
      YEAR={2021},      
      DOI={10.3389/fnins.2021.739138}  
}
  
## License
This repository is released under the Apache 2.0 license.  

