
#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT ${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-e${NORM}  --${BOLD}exp_name${NORM}. If not set, it is asked. Default is ${BOLD}local_test${NORM}."
  echo "${REV}-c${NORM}  --${BOLD}n_cpu${NORM}. Default is ${BOLD}(# of available CPUs - 1)${NORM}."
  echo "${REV}-a${NORM} + --${BOLD}approach${NORM}. Default is ${BOLD}hardcoded value${NORM}."
  echo "${REV}-g${NORM}  --${BOLD}gpu_id${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-r${NORM}  --${BOLD}results_path${NORM}. Default is ${BOLD}../hf-project/results${NORM}."
  echo "${REV}-b${NORM} + --${BOLD}nc_base${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-i${NORM} + --${BOLD}nc_incr${NORM}. Default is ${BOLD}2${NORM}."
  echo "${REV}-s${NORM} + --${BOLD}stop_task${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-t${NORM} + --${BOLD}num_tasks${NORM}. Default is ${BOLD}stop_task${NORM}."
  echo "${REV}-k${NORM}  --${BOLD}seed${NORM}. Default is ${BOLD}a set of hardocoded seeds${NORM}."
  echo "${REV}-f${NORM} + --${BOLD}fseed${NORM}. Default is ${BOLD}0-9${NORM}."
  echo "${REV}-S${NORM} + --${BOLD}shots${NORM}. Default is ${BOLD}-1${NORM}."
  echo "${REV}-R${NORM} + --${BOLD}scale${NORM}. Default is ${BOLD}1${NORM}."
  echo "${REV}-l${NORM}  --${BOLD}base_lambda${NORM}. Default is ${BOLD}5${NORM}."
  echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
  echo -e "Example: ${BOLD}$SCRIPT -e test -c 4 -b 10 -i 2 -s 4${NORM}"\\n
  exit 1
}

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
  HELP
fi

while getopts "a:g:r:e:c:b:i:s:t:k:l:S:f:R:h" opt; do
  case $opt in
    e)
      exp_name=$OPTARG
      ;;
    g)
      gpu_id=$OPTARG
      ;;
    a)
      approach+=("$OPTARG")
      ;;
    r)
      results_path=$OPTARG
      ;;
    c)
      n_cpu=$OPTARG
      ;;
    b)
      nc_base+=("$OPTARG")
      ;;
    i)
      nc_incr+=("$OPTARG")
      ;;
    s)
      stop_task+=("$OPTARG")
      ;;
    t)
      num_tasks+=("$OPTARG")
      ;;
    k)
      seed=$OPTARG
      ;;
    l)
      base_lambda=$OPTARG
      ;;
    S)
      shots+=("$OPTARG")
      ;;
    f)
      fseed+=("$OPTARG")
      ;;    
    R)
      scale+=("$OPTARG")
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

if [ -z "$exp_name" ]; then
    read -p "Please, set the experiment name [local_test]: " exp_name
    if [ -z "$exp_name" ]; then
        echo "Using default experiment name local_test."
        exp_name="local_test"
    fi
fi

if [ -z "$nc_base" ]; then
    nc_base=(0)
fi

if [ -z "$nc_incr" ]; then
	nc_incr=(2)
fi

if [ -z "$stop_task" ]; then
	stop_task=(0)
fi

if [ -z "$num_tasks" ]; then
	num_tasks=($stop_task)
fi

if [ -z "$base_lambda" ]; then
	base_lambda=5
fi

if [ -z "$shots" ]; then
	shots=(-1)
fi
if [ -z "$scale" ]; then
	scale=(1)
fi
# if [ -z "$fseed" ]; then
# 	fseed=0
# fi
exp_name="${exp_name}_base${nc_base}_incr${nc_incr}_stop${stop_task}"

campaign_ts=$(date +"%s")
if [ -z "$seed" ]; then
	exp_dir="exp_logs/exp_${campaign_ts}_${exp_name}"
else
	exp_dir="exp_logs/exp_${campaign_ts}_${exp_name}_seed${seed}"
fi

mkdir -p $exp_dir

command_file="$exp_dir/commands.log"

nomemapprs=("scratch" "finetuning" "joint" "freezing" "lwf")
memapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil" "bicp")

nomomlrapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil" "scratch" "joint" "freezing" "lwf" "bicp")
momlrapprs=("finetuning")

hdrnets=("Lopez17CNN" "Lopez17RNN" "Lopez17CNNRNN")
loadnets=("Wang17")
mmnets=("Aceto19MIMETIC")

if [ -z "$seed" ]; then
	seeds=(0 1 2 3 4 5 6 7 8 9)
else
	seeds=(${seed[@]})
fi

if [ -z "$fseed" ]; then
	fseeds=(0 1 2 3 4 5 6 7 8 9)
else
	fseeds=(${fseed[@]})
fi

if [ -z "$approach" ]; then
	approaches=("finetuning" "freezing" "lwf")
else
	approaches=(${approach[@]})
fi

if [ -z "$results_path" ]; then
	results_path="../hf-project/results"
fi

if [ -z "$gpu_id" ]; then
    gpu_id=0
fi

nets=("Lopez17CNN")

script_cmd="python3 main_incremental.py"

expname_cmd="--exp-name ${exp_name}"

respath_cmd="--results-path $results_path"

dataset_cmd="--datasets mirage_generic"

hdrinput_cmd="--fields PL IAT DIR WIN --num-pkts 10"
loadinput_cmd="--num-bytes 512"
mminput_cmd="--num-bytes 512 --fields PL IAT DIR WIN --num-pkts 20"

epochs=200
if [[ "$epochs" == 2 ]]; then
    echo "WARNING: num epochs forced to 2"
fi
echo "WARNING: training from scratch is forced."
config_cmd="--gpu $gpu_id --batch-size 64 --nepochs $epochs --last-layer-analysis --save-models --gridsearch-tasks -1 --lr-patience 10 --validation 0.1"

# echo "Seeds" ${seed[@]}
# echo "Approaches" ${approaches[@]}
# echo "N Base Classes" ${nc_base[@]}
# echo "N Incr Classes" ${nc_incr[@]}
# echo "Shots" ${shots[@]}
# echo "fSeeds" ${fseeds[@]}
echo ${scale[@]}
memoryon_cmd="--num-exemplars 1000 --exemplar-selection herding"
memoryoff_cmd=""

momlron_cmd="--momentum 0.9 --lr-factor 10 --lr-min 0.0000001"
momlroff_cmd=""

for k in ${seeds[@]}; do
    seed_cmd="--seed ${k}"
    for net in ${nets[@]}; do
        network_cmd="--network ${net}"
        if [[ " ${hdrnets[*]} " =~ " ${net} " ]]; then
            input_cmd=${hdrinput_cmd}
        elif [[ " ${loadnets[*]} " =~ " ${net} " ]]; then
            input_cmd=${loadinput_cmd}
        else
            input_cmd=${mminput_cmd}
        fi
        for appr in ${approaches[@]}; do
            approach_cmd="--approach ${appr}"
            if [[ "$appr" == "bicp" ]]; then
                base_lambda_cmd="--base-lamb $base_lambda"
            elif [[ "$appr" == "lucir" ]]; then
                base_lambda_cmd="--lamb $base_lambda"
            else
                base_lambda_cmd=""
            fi
            if [[ " ${memapprs[*]} " =~ " ${appr} " ]]; then
                memory_cmd=${memoryon_cmd}
            else
                memory_cmd=${memoryoff_cmd}
            fi
            if [[ " ${momlrapprs[*]} " =~ " ${appr} " ]]; then
                momlr_cmd=${momlron_cmd}
            else
                momlr_cmd=${momlroff_cmd}
            fi
            for ncb in ${nc_base[@]}; do
                nc_base_cmd="--nc-first-task ${ncb}"
                for nci in ${nc_incr[@]}; do
                    nc_incr_cmd="--nc-incr-tasks ${nci}"
                    for sts in ${stop_task[@]}; do
                        stop_task_cmd="--stop-at-task ${sts}"
                        for nt in ${num_tasks[@]}; do
                            num_tasks_cmd="--num-tasks ${nt}"
                            for sht in ${shots[@]}; do
                                shots_cmd="--shots ${sht}"
                                for scl in ${scale[@]}; do
                                    scale_cmd="--scale ${scl}"
                                    for fsd in ${fseeds[@]}; do
                                        fseed_cmd="--fseed ${fsd}"
                                        echo "${script_cmd} ${expname_cmd} ${respath_cmd} ${dataset_cmd} ${input_cmd} ${nc_base_cmd} ${nc_incr_cmd} ${stop_task_cmd} ${num_tasks_cmd} ${config_cmd} ${network_cmd} ${approach_cmd} ${memory_cmd} ${momlr_cmd} ${seed_cmd} ${base_lambda_cmd} ${shots_cmd} ${fseed_cmd} ${scale_cmd}" >> "${command_file}"
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

echo "OK"

xargs --arg-file $command_file --max-proc $n_cpu --replace --verbose sh -c "{}" 2> $exp_dir/logger.log
