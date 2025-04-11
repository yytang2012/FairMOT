cd src
python train.py mot --gpu 0 --exp_id mot20_ft_mix_dla34 --load_model ../models/fairmot_dla34.pth --num_epochs 20 --lr_step 15 --print_iter 10 --data_cfg ../src/lib/cfg/mot20.json
cd ..