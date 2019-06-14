export CUDA_VISIBLE_DEVICES=6
nohup python ./bin/dump_weights.py \
	--save_dir ./domain_fine_tune_iter8/checkpoint\
        --outfile='./checkpoint/weights.hdf5' 2>&1 &

