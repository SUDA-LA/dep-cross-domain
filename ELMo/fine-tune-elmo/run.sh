export CUDA_VISIBLE_DEVICES=5,6,7
nohup python -u /data/xpeng/elmo/Train_ELMo/bilm-tf/train-bilm-tf-new/bin/restart.py \
	--train_prefix='/data/xpeng/elmo/Train_ELMo/bilm-tf/train-bilm-tf-new/experiment/fine-tune-giga1100-elmo/4domain/data/4domain-trainFile/train_elmo_data.file.*'\
	--vocab_file  '/data/xpeng/elmo/Train_ELMo/bilm-tf/elmo-chinese-model/dict.train'\
	--save_dir '/data/xpeng/elmo/Train_ELMo/bilm-tf/train-bilm-tf-new/experiment/fine-tune-giga1100-elmo/4domain/4domain_fine_tune_iter8/checkpoint'\
        > /data/xpeng/elmo/Train_ELMo/bilm-tf/train-bilm-tf-new/experiment/fine-tune-giga1100-elmo/4domain/4domain_fine_tune_iter8/log.4domain-fine-tune-giga1100-model-iter8 2>&1 &
