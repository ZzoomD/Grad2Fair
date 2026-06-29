# bail_gcn
python train_grad2fair.py --hid_dim 16 --encoder gcn --dataset bail --seed_num 5 --epochs 100 --upweight_epochs 500 --alpha 20 --lr 1e-2 --run_type grad2fair

# # bail_gin
# python train_grad2fair.py --hid_dim 16 --encoder gin --dataset bail --seed_num 5 --epochs 600 --upweight_epochs 500 --alpha 19 --lr 1e-2 --run_type grad2fair

# # credit_gcn
# python train_grad2fair.py --hid_dim 16 --encoder gcn --dataset credit --seed_num 5 --epochs 700 --upweight_epochs 600 --alpha 9 --lr 1e-2 --run_type grad2fair --enable_shortcut False

# # credit_gin
# python train_grad2fair.py --hid_dim 16 --encoder gin --dataset credit --seed_num 5 --epochs 100 --upweight_epochs 500 --alpha 5 --lr 1e-3 --run_type grad2fair --enable_shortcut False

# # pokec_z_gcn
# python train_grad2fair.py --hid_dim 16 --encoder gcn --dataset pokec_z --seed_num 5 --epochs 200 --upweight_epochs 600 --alpha 11 --lr 1e-3 --run_type grad2fair

# # pokec_z_gin
# python train_grad2fair.py --hid_dim 16 --encoder gin --dataset pokec_z --seed_num 5 --epochs 450 --upweight_epochs 500 --alpha 7 --lr 1e-3 --run_type grad2fair

# # pokec_n_gcn
# python train_grad2fair.py --hid_dim 16 --encoder gcn --dataset pokec_n --seed_num 5 --epochs 100 --upweight_epochs 600 --alpha 15 --lr 1e-3 --run_type grad2fair --enable_shortcut False

# # pokec_n_gin
# python train_grad2fair.py --hid_dim 16 --encoder gin --dataset pokec_n --seed_num 5 --epochs 100 --upweight_epochs 600 --alpha 1 --lr 1e-3 --run_type grad2fair --enable_shortcut False
