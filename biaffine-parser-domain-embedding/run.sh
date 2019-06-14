exe=../driver/Train-corpus-weighting.py
nohup python3 -u $exe --config_file=config.cfg  > log.train 2>&1 &

