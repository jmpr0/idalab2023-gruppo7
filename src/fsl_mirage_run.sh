exp_name=$1

approach=$2
n_shots=$3
test_queries=$4

nc_train=$5
nc_val=$6
nc_test=$7

network=$8
scale=$9

epoch=$10

# TODO: move in start_experiment.sh
# net_param="--network=Wang17 --num_bytes=576"
#net_param="--network=Lopez17CNNRNN --num_pkts=20 --fields PL IAT DIR WIN"
net_param="--network=$network --num_pkts=10 --fields PL IAT DIR WIN --scale=$scale"
dataset="--dataset=mirage_generic --nc-train=$nc_train --nc-val=$nc_val --nc-test=$nc_test"
exp_param="--exp-name=$exp_name --max_epochs=$epoch --min_epochs=1 --lr=0.0001 --acc_grad_batches=10" # epoch_lenght = acc_grad_batches*meta-batches-per-epoch
fsl_param="--train_shots=$n_shots --train_ways=8 --train_queries=$n_shots --test_shots=$n_shots --test_ways=8 --test_queries=$test_queries"
es_param="--monitor=valid_accuracy --min_delta=0.001 --patience=20 --mode=max"

# Specific appr args here
if [ $approach == "maml" ]
then
    cmd="python3 main_meta.py --approach=maml --adaptation_steps=8 --adaptation_lr=0.01 --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd

elif [ $approach == "protonet" ]
then
    cmd="python3 main_meta.py --approach=protonet --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd

elif [ $approach == "metaoptnet" ]
then 
    cmd="python3 main_meta.py --approach=metaoptnet --svm_max_iters=15 --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd

elif [ $approach == "matchingnet" ]
then 
    cmd="python3 main_meta.py --approach=matchingnet --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd

elif [ $approach == "anil" ]
then
    cmd="python3 main_meta.py --approach=anil --adaptation_steps=8 --adaptation_lr=0.01 --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd

elif [ $approach == "relationnet" ]
then
    cmd="python3 main_meta.py --approach=relationnet --default_root_dir=../hf-project/results_FSL/${1} --seed=0 ${net_param} ${dataset} ${exp_param} ${fsl_param} ${es_param}"
    echo $cmd
    eval $cmd
else
    echo "accepted arguments are: maml, protonet, matchingnet, anil, relationnet and metaoptnet"
fi
