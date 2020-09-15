# Arbitrary-Style-Transfer-via-Multi-Adaptation-Network
Yingying Deng, Fan Tang, Weiming Dong, Wen Sun, Feiyue Huang, Changsheng Xu

## Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

## TEST 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1kUUNROxNmDroDuWl22JDlbN3vJBNYFZy/view?usp=sharing),  [decoder](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [MA_module](x)   <br> 
 
` `` 
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
` `` 
## TRAIN  
Traing set is WikiArt collected from https://www.wikiart.org/  <br>  
Test set is COCO2014  <br>  
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
