#net_param="--network=Wang17 --num_bytes=576"
#net_param="--network=Lopez17CNNRNN --num_pkts=20 --fields PL IAT DIR WIN"
net_param="--network=Lopez17CNN --num_pkts=20 --fields PL IAT DIR WIN"
#net_param="--network=Aceto19MIMETIC --num_bytes=576 --num_pkts=20 --fields PL IAT DIR WIN"
dataset="--dataset=iot23"
exp_param="--default_root_dir=../experiments/${1} --seed=42 --cleanup --gpus=0"
ml_param="--max_epochs=100 --min_epochs=1 --acc_grad_batches=12" # --nc-train=9 --nc-val=0 --nc-test=4" # --apply_fce" # epoch_lenght = acc_grad_batches*meta-batches-per-epoch
fsl_param="--train_shots=5 --train_ways=4 --train_queries=5 --test_shots=5 --test_ways=4 --test_queries=5"
es_param="--monitor=valid_accuracy --min_delta=0.01 --patience=15 --mode=max --double-monitor"
lr_param="--lr=0.0001 --lr-strat=lrop" #--scheduler_patience=10 --scheduler_decay=0.1  --t0=10 --eta-min=1e-5

# Specific appr args here
if [ "$1" == "maml" ]
then
    cmd="python3 src/main_meta.py --approach=maml --adaptation_steps=8 --adaptation_lr=0.01 ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd

elif [ "$1" == "proto_net" ]
then
    cmd="python3 src/main_meta.py --approach=proto_net ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd
            
elif [ "$1" == "metaoptnet" ]
then 
    cmd="python3 src/main_meta.py --approach=metaoptnet --svm_max_iters=15 ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd

elif [ "$1" == "matching_net" ]
then 
    cmd="python3 src/main_meta.py --approach=matching_net ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd

elif [ "$1" == "anil" ]
then
    cmd="python3 src/main_meta.py --approach=anil --adaptation_steps=8 --adaptation_lr=0.01 ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd

elif [ "$1" == "relation_net" ]
then
    cmd="python3 src/main_meta.py --approach=relation_net ${exp_param} ${net_param} ${dataset} ${ml_param} ${fsl_param} ${es_param} ${lr_param}"
    echo $cmd
    eval $cmd
else
    echo "accepted arguments are: maml, proto_net, matching_net, anil, relation_net and metaoptnet"
fi
