# A simplified version of IE-IQA
This version is based on the RBID dataset, which is small enough to re-train. Other datasets have similar performance.


## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   The model pre-trained on RBID is '**IEIQA_rbid.pt**'. The dataset are randomly splitted several times during training, and the released model is obtained from one split. Please run '**test_example_of_rbid.py**' to make both cross-dataset and intra-dataset.  The model file '**my_efficientnet.py**' and '**my_efficientnet_ie_feature.py**' is modified from open accessed source code of [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch). 
   
   
## For train:  
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat' files. Only need to run '**data_preparation_example.py**' once for each dataset.
   
### 2. Load pre-trained weight for test  
   Please run '**train_example_of_rbid.py**' to train the model. The training example can be found in '**run test_example_of_rbid.ipynb**'. The model is based on the ImageNet pre-traind weight of '**efficientnet-b0.pth**', which is obtained from [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch). 
   

## If you like this work, please cite:

{   
      AUTHOR={Song, Tianshu and Li, Leida and Zhu, Hancheng and Qian, Jiansheng},  
      TITLE={IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment},     
      JOURNAL={Frontiers in Neuroscience},     
      VOLUME={15},         
      YEAR={2021},      
      DOI={10.3389/fnins.2021.739138}  
}



