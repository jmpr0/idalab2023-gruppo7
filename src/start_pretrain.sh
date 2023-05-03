results_path="--default_root_dir ../hf-project/results_PT/"
saved_weights_path="--saved_weights_path ../../saved_weights/"
dataset="--dataset mirage_generic"
input="--num_pkts 10 --fields PL IAT DIR WIN"
net="--network ShallowAutoencoder"
exp_param="--max_epochs 100 --min_epochs 1 --lr 0.0001"
es_param="--monitor valid_loss --min_delta 0.001 --mode min"
lrs_param="--scheduler_patience -1 --scheduler_decay 0.1"
seed="--seed 0"

cmd="python3 main_tl.py ${results_path} ${saved_weights_path} ${dataset} ${input} ${net} ${exp_param} ${es_param} ${lrs_param} ${seed}"
echo $cmd
eval $cmd