export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
python ./bin/train_elmo.py \
	--train_prefix='/data1/yli/giga-train-data/split/sens.train.split.*' \
	--vocab_file  '/data1/yli/giga-train-data/dict.train'\
	--save_dir /data1/yli/elmo-chinese-model/checkpoint
