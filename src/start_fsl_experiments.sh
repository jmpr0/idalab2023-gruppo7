
#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT ${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-e${NORM}  --${BOLD}exp_name${NORM}. If not set, it is asked. Default is ${BOLD}test${NORM}."
  echo "${REV}-c${NORM}  --${BOLD}n_cpu${NORM}. Default is ${BOLD}(# of available CPUs - 1)${NORM}."
  echo "${REV}-a${NORM} + --${BOLD}approach${NORM}. Default is ${BOLD}protonet${NORM}."
  echo "${REV}-g${NORM}  --${BOLD}gpus${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-r${NORM}  --${BOLD}results_path${NORM}. Default is ${BOLD}../hf-project/results_FSL${NORM}."
  echo "${REV}-s${NORM} + --${BOLD}n_shots${NORM}. Default is ${BOLD}15${NORM}."
  echo "${REV}-w${NORM} + --${BOLD}n_ways${NORM}. Default is ${BOLD}8${NORM}."
  echo "${REV}-q${NORM} + --${BOLD}test_queries${NORM}. Default is ${BOLD}50${NORM}."
  echo "${REV}-t${NORM} + --${BOLD}nc_train${NORM}. Default is ${BOLD}24${NORM}."
  echo "${REV}-v${NORM}  --${BOLD}nc_val${NORM}. Default is ${BOLD}8${NORM}."
  echo "${REV}-T${NORM}  --${BOLD}nc_test${NORM}. Default is ${BOLD}8${NORM}."
  echo "${REV}-k${NORM} + --${BOLD}seed${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-n${NORM} + --${BOLD}network${NORM}. Default is ${BOLD}Lopez17CNN${NORM}."
  echo "${REV}-S${NORM} + --${BOLD}scale${NORM}. Default is ${BOLD}1${NORM}."
  echo "${REV}-l${NORM} + --${BOLD}loss_factor${NORM}. Default is ${BOLD}1${NORM}."
  echo "${REV}-E${NORM}  --${BOLD}epoch${NORM}. Default is ${BOLD}100${NORM}."
  echo "${REV}-p${NORM} + --${BOLD}patience${NORM}. Default is ${BOLD}20${NORM}."
  echo "${REV}-A${NORM} + --${BOLD}augmentation${NORM}. Default is ${BOLD}n${NORM}."
  echo "${REV}-d${NORM}  --${BOLD}dataset${NORM}. Default is ${BOLD}mirage_generic${NORM}."
  echo "${REV}-b${NORM} + --${BOLD}num_bytes${NORM}. Default is ${BOLD}512${NORM}."
  echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
  echo -e "Example: ${BOLD}$SCRIPT -e local_test -s 30 -q 50 -t 24 -v 8 -T 8 -E 100${NORM}"\\n
  exit 1
}

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
  HELP
fi

while getopts "e:c:a:g:r:s:w:q:t:v:T:k:n:S:l:E:p:A:d:b:h" opt; do
  case $opt in
    e)
      exp_name=$OPTARG
      ;;
    c)
      n_cpu=$OPTARG
      ;;
    a)
      approach+=("$OPTARG")
      ;;
    g)
      gpus=$OPTARG
      ;;
    r)
      results_path=$OPTARG
      ;;
    s)
      n_shots+=("$OPTARG")
      ;;
    w)
      n_ways+=("$OPTARG")
      ;;
    q)
      test_queries+=("$OPTARG")
      ;;
    t)
      nc_train+=("$OPTARG")
      ;;
    v)
      nc_val=$OPTARG
      ;;
    T)
      nc_test=$OPTARG
      ;;
    k)
      seed+=("$OPTARG")
      ;;
    n)
      network+=("$OPTARG")
      ;;
    S)
      scale+=("$OPTARG")
      ;;
    l)
      loss_factor+=("$OPTARG")
      ;;
    E)
      epoch=$OPTARG
      ;;
    p)
      patience+=("$OPTARG")
      ;;
    A)
      augs+=("$OPTARG")
      ;;
    d)
      dataset=$OPTARG
      ;;
    b)
      num_bytes+=("$OPTARG")
      ;;
    h)
      HELP
      ;;
    \?)
      echo -e \\n"Option -${BOLD}$OPTARG${NORM} not allowed."
      HELP
      ;;
  esac
done

set_default() { echo $1; }

# Default argument values
exp_name=$(set_default $exp_name "test")
approach=$(set_default $approach "proto_net")
gpu_id=$(set_default $gpus "0")
results_path=$(set_default $results_path "../hf-project/results_FSL")
n_shots=$(set_default $n_shots "15")
n_ways=$(set_default $n_ways "8")
test_queries=$(set_default $test_queries "50")
nc_train=$(set_default $nc_train "24")
nc_val=$(set_default $nc_val "8")
nc_test=$(set_default $nc_test "8")
seed=$(set_default $seed "0")
network=$(set_default $network "Lopez17CNN")
scale=$(set_default $scale "1.0")
loss_factor=$(set_default $loss_factor "1")
epoch=$(set_default $epoch "100")
patience=$(set_default $patience "20")
augs=$(set_default $augs "n")
dataset=$(set_default $dataset "mirage_generic")
num_bytes=$(set_default $num_bytes "512")

echo ${seed[@]}
echo ${network[@]}
echo ${approach[@]}
echo ${n_shots[@]}
echo ${n_ways[@]}
echo ${test_queries[@]}
echo ${nc_train[@]}
echo ${patience[@]}
echo ${scale[@]}
echo ${loss_factor[@]}
echo ${augs[@]}
echo ${dataset[@]}
echo ${num_bytes[@]}

campaign_ts=$(date +"%s")
exp_dir="fsl_exp_logs/exp_${exp_name}_seed${seed}_${campaign_ts}"

mkdir -p $exp_dir

command_file="$exp_dir/commands.log"

script_cmd="python3 main_meta.py --return_concat"

#dataset="mirage_generic"
dataset_cmd="--dataset $dataset --nc-val $nc_val --nc-test $nc_test"

hdrnets=("Lopez17CNN" "Lopez17RNN" "Lopez17CNNRNN" "Jorgensen22MLP" "Lopez17CNNDrop" "Lopez17CNNMLP")
loadnets=("Wang17" "Aceto19CNN")
mmnets=("Aceto19MIMETIC")
#
hdrinput_cmd="--fields PL IAT DIR WIN --num_pkts 10"
# loadinput_cmd="--num_bytes 512"
# mminput_cmd="--num_bytes 512 --fields PL IAT DIR WIN --num_pkts 10"

config_cmd="--gpus $gpus --max_epochs $epoch --min_epochs 1 --acc_grad_batches 10 --test_ways $nc_test"
es_cmd="--monitor valid_accuracy --min_delta 0.001 --mode max --double-monitor --cleanup"
lr_cmd="--lr 0.0001 --lr-strat none" #--scheduler_patience 10 --scheduler_decay  --t0 10 --eta-min 1e-5

optapprs=("maml" "anil")
svmapprs=("metaoptnet")
fceapprs=("matching_net")
#
opton_args="--adaptation_steps=8 --adaptation_lr=0.01 --first_order"
svmon_args="--svm_max_iters=15"

cnt=0
for k in ${seed[@]}; do
	seed_cmd="--seed ${k}"
	for net in ${network[@]}; do
    for nb in ${num_bytes[@]}; do
      network_cmd="--network ${net}"
      if [[ " ${hdrnets[*]} " =~ " ${net} " ]]; then
        input_cmd=${hdrinput_cmd}
        fce_args="--apply_fce"
      elif [[ " ${loadnets[*]} " =~ " ${net} " ]]; then
        input_cmd="--num_bytes $nb"
        fce_args="--apply_fce"
      else
        input_cmd="--num_bytes $nb --fields PL IAT DIR WIN --num_pkts 10"
        fce_args=""
      fi
      for appr in ${approach[@]}; do
        approach_cmd="--approach ${appr}"
        if [[ " ${optapprs[*]} " =~ " ${appr} " ]]; then
          opt_cmd=${opton_args}
        else
          ff=""
        fi
        if [[ " ${svmapprs[*]} " =~ " ${appr} " ]]; then
          svm_cmd=${svmon_args}
        else
          svm_cmd=""
        fi
        if [[ " ${fceapprs[*]} " =~ " ${appr} " ]]; then
          fce_cmd=${fce_args}
        else
          fce_cmd=""
        fi
        for sht in ${n_shots[@]}; do
          sht_cmd="--train_shots $sht --train_queries $sht --test_shots $sht"
          for way in ${n_ways[@]}; do
            way_cmd="--train_ways $way"
            for teq in ${test_queries[@]}; do
              teq_cmd="--test_queries $teq"
              for ctr in ${nc_train[@]}; do
                ctr_cmd="--nc-train $ctr"
                for pat in ${patience[@]}; do
                  pat_cmd="--patience $pat"
                  for lf in ${loss_factor[@]}; do
                    lf_cmd="--loss-factor $lf"
                    for aug in ${augs[@]}; do
                      aug_cmd="--augs $aug"
                      for scl in ${scale[@]}; do
                        scl_cmd="--scale $scl"
                        timestamp=$(date +%s)
                        rootdir_cmd="--default_root_dir ${results_path}/${dataset}_${appr}_ctr_${ctr}_sht_${sht}_way_${way}_epc_${epoch}_pat_${pat}"
                        rootdir_cmd+="_net_$(echo "$net" | tr '[:upper:]' '[:lower:]')_scl_$(echo "$scl" | sed 's|\.|-|g')_${timestamp}"
                        t=$(( cnt % (n_cpu * 5)  ))
                        echo "sleep $t && ${script_cmd} ${dataset_cmd} ${input_cmd} ${svm_cmd} ${seed_cmd} ${sht_cmd} ${way_cmd} ${teq_cmd} ${config_cmd} ${network_cmd}"\
                            " ${approach_cmd} ${opt_cmd} ${fce_cmd} ${ctr_cmd} ${pat_cmd} ${lf_cmd} ${aug_cmd} ${scl_cmd} ${es_cmd} ${lr_cmd} ${rootdir_cmd}" >> "${command_file}"
                        cnt=$(( cnt + 5 ))
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
	done
done

xargs --arg-file $command_file --max-proc $n_cpu --replace --verbose sh -c "{}" 2> $exp_dir/logger.log

# if [[ $(basename "$n_cpu") == processors* ]]; then
# 	processors=$n_cpu
# 	flock $processors -c "dos2unix $processors"
# 	echo "Using the processors in $processors."
# else
# 	if [ -z "$n_cpu" ]; then
# 		read -p "Please, set the number of processor(s) [ncpus-1]: " n_cpu
# 		if [ -z "$n_cpu" ]; then
# 			echo "All-processes execution (default n_cpu is ncpus-1)."
# 			n_cpu=$(( $(cat /proc/cpuinfo | grep processor | wc -l) - 1 ))
# 		fi
# 	fi
# 	echo "Using $n_cpu processor(s)."
# 	sleep 3
 
#   # TODO: change the "processors_${campaign_ts}" filename to "processors_${n_cpu}" when no experiment is running

# 	# storing of per cpu mask for the taskset command.
# 	# N.B. the first processor is not used because for experiments, but it should be used for support ops
# 	processors="$exp_dir/processors_${n_cpu}.dat"
# 	if [ ! -f "$processors" ]; then
# 		for cpu in $(seq 1 ${n_cpu}); do
# 			echo $(echo "obase=16; $(( 2 ** ${cpu} ))" | bc) >> $processors
# 		done
# 	fi
# fi

# n_exp=$(cat $command_file | wc -l)
# for exp_id in $(seq $n_exp); do
# 	./execute.sh $processors $command_file true &
# 	pids[${i}]=$!
# 	taskset 1 sleep 30
# done

# for pid in "${pids[@]}"; do
# 	wait $pid
# done

# echo "Computing metrics..."
# python3 compute_metrics.py $expname_cmd $respath_cmd --no --njobs 10

# echo "Plotting metrics..."
# python3 compute_metrics.py $expname_cmd $respath_cmd --metric all --analysis all --no --njobs 10
