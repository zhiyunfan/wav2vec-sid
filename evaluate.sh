model_path=./exp/wav2vev2_small_finetune_VOX1_amsoftmax_2/checkpoint_last.pt
data_path=/mnt/lustre/xushuang2/zyfan/data/vox/test_list.txt

CUDA_VISIBLE_DEVICES=7 python test_speaker.py $data_path  \
--task audio_pretraining_sid --path $model_path --criterion  classification_amsoftmax  \
