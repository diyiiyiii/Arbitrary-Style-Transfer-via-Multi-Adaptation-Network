# Arbitrary-Style-Transfer-via-Multi-Adaptation-Network
python test.py  --content_dir input/content/ --style_dir input/style/   --decoder models//decoder_iter_  --atte models/   --output test-vincent2

CUDA_VISIBLE_DEVICES=0 python train.py --style_dir ../../datasets/Images/ --content_dir ../../datasets/train2014 --save_dir models/model-v4/ --batch_size 8
