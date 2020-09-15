# Arbitrary-Style-Transfer-via-Multi-Adaptation-Network
Yingying Deng, Fan Tang, Weiming Dong, Wen Sun, Feiyue Huang, Changsheng Xu  <br>
Paper Link [pdf](https://arxiv.org/abs/2005.13219)

## Framework
![image](http://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/framework/framework.png)


## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* PIL, numpy, scipy
* tqdm  <br> 

### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1kUUNROxNmDroDuWl22JDlbN3vJBNYFZy/view?usp=sharing),  [decoder](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [MA_module](x)   <br> 
Please download them and put them into the floder  ./model/  <br> 
`
python test.py  --content_dir input/content/ --style_dir input/style/    --output out
`
### Training  
Traing set is WikiArt collected from [WIKIART](https://www.wikiart.org/)  <br>  
Testing set is COCO2014  <br>  
`
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
`
### Reference
If you use our work in your research, please cite us using the following BibTeX entry. <br> 
```
@inproceedings{deng:2020:arbitrary,
  title={Arbitrary Style Transfer via Multi-Adaptation Network},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Sun, Wen and Huang, Feiyue and Xu, Changsheng},
  booktitle={Acm International Conference on Multimedia},
  year={2020},
 publisher = {ACM},
}
```
