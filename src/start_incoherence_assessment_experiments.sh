
#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT ${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-e${NORM}  --Sets the value for option ${BOLD}exp_name${NORM}. If not set, it is asked. Default is ${BOLD}local_test${NORM}."
  echo "${REV}-c${NORM}  --Sets the value for option ${BOLD}n_cpu${NORM}. Default is ${BOLD}(# of available CPUs - 1)${NORM}."
  echo "${REV}-b${NORM}  --Sets the value for option ${BOLD}nc_base${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-i${NORM}  --Sets the value for option ${BOLD}nc_incr${NORM}. Default is ${BOLD}2${NORM}."
  echo "${REV}-s${NORM}  --Sets the value for option ${BOLD}stop_task${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-t${NORM}  --Sets the value for option ${BOLD}num_tasks${NORM}. Default is ${BOLD}stop_task${NORM}."
  echo "${REV}-k${NORM}  --Sets the value for option ${BOLD}seed${NORM}. Default is ${BOLD}a set of hardocoded seeds${NORM}."
  echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
  echo -e "Example: ${BOLD}$SCRIPT -e test -c 4 -b 10 -i 2 -s 4${NORM}"\\n
  exit 1
}

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
  HELP
fi

while getopts "e:c:b:i:s:t:k:h" opt; do
  case $opt in
    e)
      exp_name=$OPTARG
      ;;
    c)
      n_cpu=$OPTARG
      ;;
    b)
      nc_base=$OPTARG
      ;;
    i)
      nc_incr=$OPTARG
      ;;
    s)
      stop_task=$OPTARG
      ;;
    t)
      num_tasks=$OPTARG
      ;;
    k)
      seed=$OPTARG
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
	nc_base=0
fi

if [ -z "$nc_incr" ]; then
	nc_incr=2
fi

if [ -z "$stop_task" ]; then
	stop_task=0
fi

if [ -z "$num_tasks" ]; then
	num_tasks=$stop_task
fi

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
memapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil")

nomomlrapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil" "scratch" "joint" "freezing" "lwf")
momlrapprs=("finetuning")

hdrnets=("Lopez17CNN" "Lopez17RNN" "Lopez17CNNRNN")
loadnets=("Wang17")
mmnets=("Aceto19MIMETIC")
if [ -z "$seed" ]; then
	seeds=(0 1 2 3 4 5 6 7 8 9)
	# OVERRIDE default seeds
	# seeds=(5 6 7 8 9)
else
	seeds=($seed)
fi

echo "OVERRIDE"
nomemapprs=()
memapprs=("bic")
hdrnets=("Lopez17CNN")
loadnets=()
mmnets=()

script_cmd="python3 main_incremental.py"

expname_cmd="--exp-name ${exp_name}"

respath_cmd="--results-path ../hf-project/results"

dataset_cmd="--datasets mirage_generic"

hdrinput_cmd="--fields PL IAT DIR WIN --num-pkts 10"
loadinput_cmd="--num-bytes 512"
mminput_cmd="--num-bytes 512 --fields PL IAT DIR WIN --num-pkts 10"

echo "WARNING:"

config_cmd="--batch-size 64 --nepochs 200 --last-layer-analysis --save-models --gridsearch-tasks -1 --lr-patience 20"
nc_base_cmd="--nc-first-task ${nc_base}"
nc_incr_cmd="--nc-incr-tasks ${nc_incr}"
stop_task_cmd="--stop-at-task ${stop_task}"
num_tasks_cmd="--num-tasks ${num_tasks}"

memoryon_cmd="--num-exemplars 1000 --exemplar-selection herding"
memoryoff_cmd=""

momlron_cmd="--momentum 0.9 --lr-factor 10 --lr-min 0.0000001"
momlroff_cmd=""

# t=1
for k in ${seeds[@]}; do
	seed_cmd="--seed ${k}"
	for net in ${hdrnets[@]} ${loadnets[@]}; do
		network_cmd="--network ${net}"
		if [[ " ${hdrnets[*]} " =~ " ${net} " ]]; then
			input_cmd=${hdrinput_cmd}
		elif [[ " ${loadnets[*]} " =~ " ${net} " ]]; then
			input_cmd=${loadinput_cmd}
		else
			input_cmd=${mminput_cmd}
		fi
		for appr in ${memapprs[@]} ${nomemapprs[@]}; do
			approach_cmd="--approach ${appr}"
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
			echo "${script_cmd} ${expname_cmd} ${respath_cmd} ${dataset_cmd} ${input_cmd} ${nc_base_cmd} ${nc_incr_cmd} ${stop_task_cmd} ${num_tasks_cmd} ${config_cmd} ${network_cmd} ${approach_cmd} ${memory_cmd} ${momlr_cmd} ${seed_cmd}" >> "${command_file}"
			# echo "stress -c 1 -t ${t}s" >> "${command_file}"
			# t=$(( $t + 1 ))
		done
	done
done

if [[ $(basename "$n_cpu") == processors* ]]; then
	processors=$n_cpu
	flock $processors -c "dos2unix $processors"
	echo "Using the processors in $processors."
else
	if [ -z "$n_cpu" ]; then
		read -p "Please, set the number of processor(s) [ncpus-1]: " n_cpu
		if [ -z "$n_cpu" ]; then
			echo "All-processes execution (default n_cpu is ncpus-1)."
			n_cpu=$(( $(cat /proc/cpuinfo | grep processor | wc -l) - 1 ))
		fi
	fi
	echo "Using $n_cpu processor(s)."
	sleep 3
 
  # TODO: change the "processors_${campaign_ts}" filename to "processors_${n_cpu}" when no experiment is running

	# storing of per cpu mask for the taskset command.
	# N.B. the first processor is not used because for experiments, but it should be used for support ops
	processors="$exp_dir/processors_${n_cpu}.dat"
	if [ ! -f "$processors" ]; then
		for cpu in $(seq 1 ${n_cpu}); do
			echo $(echo "obase=16; $(( 2 ** ${cpu} ))" | bc) >> $processors
		done
	fi
fi

n_exp=$(cat $command_file | wc -l)
for exp_id in $(seq $n_exp); do
	./execute.sh $processors $command_file &
	pids[${i}]=$!
	taskset 1 sleep 30
done

# for pid in "${pids[@]}"; do
# 	wait $pid
# done

# echo "Computing metrics..."
# python3 compute_metrics.py $expname_cmd $respath_cmd --no --njobs 10

# echo "Plotting metrics..."
# python3 compute_metrics.py $expname_cmd $respath_cmd --metric all --analysis all --no --njobs 10
