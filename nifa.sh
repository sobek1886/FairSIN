# None of the attemps on running the model on DBLP dataset worked

### FairSIN not poisoned
# pokec_z
python in-train.py --dataset='pokec_z' --encoder='GCN'  --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='pokec_z' --encoder='GCN'  --delta=1 --hidden=128

# pokec_n
python in-train.py --dataset='pokec_n' --encoder='GCN'  --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='pokec_n' --encoder='GCN'  --delta=1 --hidden=128

# DBLP
python in-train.py --dataset='dblp' --encoder='GCN'  --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='dblp' --encoder='GCN' --hidden=128

### FairSIN poisoned
# pokec_z poisoned
python in-train.py --dataset='pokec_z_poisoned' --encoder='GCN'  --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='pokec_z_poisoned' --encoder='GCN'  --delta=1 --runs=30 --hidden=128

# pokec_n poisoned
python in-train.py --dataset='pokec_n_poisoned' --encoder='GCN'  --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='pokec_n_poisoned' --encoder='GCN'  --delta=1 --runs=30 --hidden=128

# DBLP poisoned
python in-train.py --dataset='dblp_poisoned' --encoder='GCN' --hidden=128 --d_lr=0.001 --c_lr=0.001 --e_lr=0.001 --m_lr=0.001 --delta=4

python in-train.py --dataset='dblp_poisoned' --encoder='GCN' --hidden=128