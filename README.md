# IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment
This is the source code for [IE-IQA: Intelligibility Enriched Generalizable No-Reference Image Quality Assessment](https://www.frontiersin.org/articles/10.3389/fnins.2021.739138/full).![IE-IQA Framework](https://github.com/esnthere/IEIT/blob/main/framework.png)

## Dependencies and Installation
Pytorch: 1.8.1  
timm: 0.3.2  
CUDA: 10.2  

## For test:
### 1. Data preparation  
   To ensure high speed, save images and lables of each dataset with 'mat/npz' files. Only need to run '**data_preparation_example.py**' once for each dataset. Enhanced images and files for slection probability can be download from: [Enhanced images](https://pan.baidu.com/s/1vSeiH61x5TD5VIRn8NKOVw?pwd=wz98). Please download these files and put them into the same folder of code.
   
### 2. Load pre-trained weight for test  
   The models pre-trained on KonIQ-10k, SPAQ, LIVEW, RBID, CID2013 are released. The dataset are randomly splitted several times during training, and each released model is obtained from one split. The model file '**my_vision_transformer.py**' is modified from open accessed source code of [DEIT](https://github.com/facebookresearch/deit) and [TIMM](https://github.com/huggingface/pytorch-image-models/tree/main/timm). 
   
   The pre-trained models can be downloaded from: [Pre-trained models](https://pan.baidu.com/s/1IoGWFXKSi-ljYaWB6mOeaw?pwd=f4wv ). Please download these files and put them in the same folder of code and then run '**test_example_of_*dataset*.py**' to test models trained on the *dataset*. You can make both cross-dataset and intra-dataset of the model trained on KonIQ-10k.
   
   
## For train:  
The training code can be available at the 'training' folder.


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

