# Arbitrary-Style-Transfer-via-Multi-Adaptation-Network

--TEST
python test.py  --content_dir input/content/ --style_dir input/style/   --decoder models//decoder_iter_  --atte models/   --output test-vincent2

--TRAIN
Traing set is WikiArt collected from https://www.wikiart.org/
Test set is COCO2014
python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 8
