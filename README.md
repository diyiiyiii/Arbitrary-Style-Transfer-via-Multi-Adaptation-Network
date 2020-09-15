# Arbitrary-Style-Transfer-via-Multi-Adaptation-Network
Yingying Deng, Fan Tang, Weiming Dong, Wen Sun, Feiyue Huang, Changsheng Xu

## Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

## TEST 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1kUUNROxNmDroDuWl22JDlbN3vJBNYFZy/view?usp=sharing),  [decoder](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [MA_module](x)   <br> 
 
`
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
`
## TRAIN  
Traing set is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
Test set is COCO2014  <br>  
`
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
`
## Reference
If you use our work in your research, please cite us using the following BibTeX entry. <br> 
`
@article{deng2020arbitrary, <br> 
  title={Arbitrary Style Transfer via Multi-Adaptation Network}, <br> 
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Sun, Wen and Huang, Feiyue and Xu, Changsheng}, <br> 
  journal={arXiv preprint arXiv:2005.13219}, <br> 
  year={2020} <br> 
}
`
