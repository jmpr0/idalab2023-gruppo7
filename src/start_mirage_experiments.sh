
#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT ${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-d${NORM}  --Sets the value for option ${BOLD}dataset${NORM}. If not set, it is asked. Default is ${BOLD}mirage_generic${NORM}."
  echo "${REV}-g${NORM}  --Sets the value for option ${BOLD}gpu_id${NORM}. If not set, it is asked. Default is ${BOLD}0${NORM}."
  echo "${REV}-e${NORM}  --Sets the value for option ${BOLD}exp_name${NORM}. If not set, it is asked. Default is ${BOLD}local_test${NORM}."
  echo "${REV}-a${NORM} + --Sets the value for option ${BOLD}approach${NORM}. Default is the entire set of approaches."
  echo "${REV}-r${NORM}  --Sets the value for option ${BOLD}results_path${NORM}. Default is ${BOLD}../results/${NORM}$."
  echo "${REV}-u${NORM}  --Sets the value for option ${BOLD}base_model_path${NORM}. Default is ${BOLD}../results/${NORM}$."
  echo "${REV}-m${NORM} + --Sets the value for option ${BOLD}models${NORM}. Default is ${BOLD}Lopez17CNN${NORM}$."
  echo "${REV}-c${NORM}  --Sets the value for option ${BOLD}n_cpu${NORM}. Default is ${BOLD}(# of available CPUs - 1)${NORM}."
  echo "${REV}-b${NORM} + --Sets the value for option ${BOLD}nc_base${NORM}. Default is ${BOLD}0${NORM}."
  echo "${REV}-i${NORM} + --Sets the value for option ${BOLD}nc_incr${NORM}. Default is ${BOLD}2${NORM}."
  echo "${REV}-x${NORM}  --Sets the value for option ${BOLD}max_classes${NORM}. Default is ${BOLD}40${NORM}."
  echo "${REV}-s${NORM}  --Sets the value for option ${BOLD}stop_task${NORM}. Default is ${BOLD}auto${NORM}."
  echo "${REV}-S${NORM}  --Sets the value for option ${BOLD}from_scratch${NORM}. Default is ${BOLD}false${NORM}."
  echo "${REV}-t${NORM}  --Sets the value for option ${BOLD}num_tasks${NORM}. Default is ${BOLD}stop_task${NORM}."
  echo "${REV}-k${NORM} + --Sets the value for option ${BOLD}seed${NORM}. Default is ${BOLD}a set of hardocoded seeds${NORM}. Special values are: f10 (from 1 to 10)"
  echo "${REV}-l${NORM} + --Sets the value for option ${BOLD}last_class${NORM}. Default is ${BOLD}-1${NORM}. Special values are: all (from 0 to max_classes), h1 (the first half of max_classes), and h2 (the second half of max_classes)"
  echo "${REV}-o${NORM}   --Sets the value for option ${BOLD}out_feat_size${NORM}. By default it is not passed."
  echo "${REV}-M${NORM}  --Sets the value for option ${BOLD}memory_size${NORM}. Default is ${BOLD}auto${NORM}."
  echo "${REV}-f${NORM}  --Sets the value for option ${BOLD}failed_cmds${NORM}. Default is ${BOLD}None${NORM}."
  echo "${REV}-H${NORM} + --Sets the value for option ${BOLD}shots${NORM}. Default is ${BOLD}-1${NORM}."
  echo "${REV}-E${NORM} + --Sets the value for option ${BOLD}fseed${NORM}. Default is ${BOLD}-1${NORM}. Special values are: f10 (from 0 to 9)"
  echo "${REV}-C${NORM} + --Sets the value for option ${BOLD}scale${NORM}. Default is ${BOLD}1${NORM}."
  echo "${REV}-p${NORM}  --Sets the value for option ${BOLD}predict_tasks${NORM}. Default is ${BOLD}false${NORM}."
  echo "${REV}-T${NORM}  --Sets the value for option ${BOLD}statistical${NORM}. Default is ${BOLD}false${NORM}."
  echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
  echo -e "Example: ${BOLD}$SCRIPT -e test -c 4 -b 10 -i 2 -s 4${NORM}"\\n
  exit 1
}

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
  HELP
fi

while getopts "d:g:e:a:r:u:c:b:i:m:x:s:St:k:l:o:M:f:H:E:C:pTh" opt; do
  case $opt in
    d)
      dataset=$OPTARG
      ;;
    g)
      gpu_id=$OPTARG
      ;;
    e)
      exp_name=$OPTARG
      ;;
    a)
      approach+=("$OPTARG")
      ;;
    r)
      results_path=$OPTARG
      ;;
    u)
      base_model_path=$OPTARG
      ;;
    m)
	  models+=("$OPTARG")
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
    x)
      max_classes=$OPTARG
      ;;
    s)
      stop_task=$OPTARG
      ;;
    S)
      from_scratch="true"
      ;;
    t)
      num_tasks=$OPTARG
      ;;
    k)
      seed+=($OPTARG)
      ;;
    l)
      last_class+=($OPTARG)
      ;;
    o)
      out_feat_size=$OPTARG
      ;;
    M)
      memory_size=$OPTARG
      ;;
    f)
      failed_cmds+=("$OPTARG")
      ;;
    H)
      shots+=("$OPTARG")
      ;;
    E)
      fseed+=("$OPTARG")
      ;;
    C)
      scale+=("$OPTARG")
      ;;
    p)
      predict_tasks="true"
      ;;
    T)
      statistical="true"
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

if [ -z "$n_cpu" ]; then
	read -p "Please, set the number of processor(s) [ncpus-1]: " n_cpu
	if [ -z "$n_cpu" ]; then
		echo "All-processes execution (default n_cpu is ncpus-1)."
		n_cpu=$(( $(cat /proc/cpuinfo | grep processor | wc -l) - 1 ))
	fi
fi
echo "Using $n_cpu processor(s)."
sleep 3

if [ -z "$gpu_id" ]; then
	gpu_id=0
fi

if [ -z "$nc_base" ]; then
	nc_base=0
fi

if [ -z "$nc_incr" ]; then
	nc_incr=2
fi

if [ -z "$max_classes" ]; then
	max_classes=40
fi

if [ -z "$stop_task" ]; then
	stop_task='none'
elif [ -z "$num_tasks" ]; then
	num_tasks=$stop_task
fi

campaign_ts=$(date +"%s")
if [ -z "$seed" ]; then
	exp_dir="exp_logs/exp_${campaign_ts}_${exp_name}"
else
	exp_dir="exp_logs/exp_${campaign_ts}_${exp_name}_seed${seed}"
fi

mkdir -p $exp_dir

command_file="$exp_dir/commands.log"

if [ -z "$failed_cmds" ]; then

	echo "WARNING: only a subset of approaches is launched"

	nomemapprs=("scratch" "finetuning" "joint" "freezing" "lwf" "jointft" "backbonefreezing" "multiclass_classifiers")
	memapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil" "bicpp" "bicp" "wu2022" "chen2021" "multiclass_classifiers")

	nomomlrapprs=("lucir" "il2m" "icarlp" "icarlo" "icarl" "ewc" "bic" "eeil" "ssil" "scratch" "joint" "freezing" "lwf" "backbonefreezing" "bicpp" "bicp" "wu2022" "chen2021" "multiclass_classifiers")
	momlrapprs=("finetuning" "jointft")

	hdrnets=("Lopez17CNN" "Lopez17RNN" "Lopez17CNNRNN" "RandomForest" "XGB")
	loadnets=("Wang17")
    
    mlmodels=("RandomForest" "XGB")

	apprs=("finetuning" "freezing" "lwf" "jointft" "backbonefreezing" "lucir" "il2m" "icarlp" "icarl" "ewc" "bic" "eeil" "ssil" "wu2022" "chen2021" "multiclass_classifiers")
    
    if [ -z "$dataset" ]; then
		models="mirage_generic"
	fi
	
	if [ -z "$models" ]; then
		models=("Lopez17CNN")
	fi

	if [ -z "$seed" ]; then
		seeds=(1 2 3 4 5 6 7 8 9 10)
		# OVERRIDE default seeds
		# seeds=(5 6 7 8 9)
    elif [[ "$seed" == "f10" ]]; then
		seeds=$(seq 1 10)
	else
		seeds=${seed[@]}
	fi

	if [ -z "$last_class" ]; then
		last_class=-1
	elif [[ "$last_class" == "all" ]]; then
		last_class=$(seq 0 $(( max_classes - 1 )) )
	elif [[ "$last_class" == "h1" ]]; then
		last_class=$(seq 0 $(( (max_classes - 1) / 2 )) )
	elif [[ "$last_class" == "h2" ]]; then
		last_class=$(seq $(( (max_classes - 1) / 2 + 1 )) $(( max_classes - 1 )) )
	fi

	echo "last classes: ${last_class[@]}"

	if [ -z "$approach" ]; then
		approaches=${apprs[@]}
	else
		approaches=${approach[@]}
	fi

	if [ -z "$results_path" ]; then
		results_path="../results/"
	fi

	if [ -z "$from_scratch" ]; then
		from_scratch="false"
	fi

	if [ -z "$memory_size" ]; then
		memory_size="auto"
	fi

	if [ -z "$out_feat_size" ]; then
		out_feat_size_cmd=""
	else
		out_feat_size_cmd="--out-features-size $out_feat_size "
	fi
    
    if [ -z "$fseed" ]; then
		fseeds=(-1)
	elif [[ "$fseed" == "f10" ]]; then
		fseeds=$(seq 0 9)
    else
        fseeds=${fseed[@]}
	fi
    
    if [ -z "$scale" ]; then
		scale=1
	fi
    
    if [ -z "$shots" ]; then
		shots=-1
	fi
    
    if [ -z "$predict_tasks" ]; then
		predict_tasks="false"
	fi
    
    if [ -z "$statistical" ]; then
		statistical="false"
	fi

	script_cmd="python3 main_incremental.py"

	respath_cmd="--results-path ${results_path} "

	dataset_cmd="--datasets $dataset "
    
    if [[ $dataset == "mirage_generic" ]]; then
        hdrinput_cmd="--fields PL IAT DIR WIN --num-pkts 10 "
        loadinput_cmd="--num-bytes 512 "
    else  # if dataset == iot23
        hdrinput_cmd="--fields PL IAT DIR WIN --num-pkts 20 "
        loadinput_cmd="--num-bytes 576 "
    fi

	if [ -z "$base_model_path" ]; then
		base_model_path_cmd=""
	else
		base_model_path_cmd="--base-model-path $base_model_path "
	fi
    
    extra_cmds=""

	if [[ $from_scratch == "false" ]]; then
		extra_cmds="$extra_cmds --load-base-model $base_model_path_cmd "
	fi
    if [[ $predict_tasks == "true" ]]; then
		extra_cmds="$extra_cmds --predict-tasks "
	fi
    if [[ $statistical == "true" ]]; then
		extra_cmds="$extra_cmds --statistical "
	fi

	#echo "WARNING: --all-outputs forced."
	#config_cmd="--gpu ${gpu_id} --batch-size 64 --nepochs 200 --last-layer-analysis --save-models --gridsearch-tasks -1 --lr-patience 20 $extra_cmds --all-outputs"
	config_cmd="--gpu ${gpu_id} --batch-size 64 --nepochs 200 --save-models --gridsearch-tasks -1 --lr-patience 20 $extra_cmds "
    #--max-depth 3"

	if [[ $memory_size == "auto" ]]; then
		memoryon_cmd="--num-exemplars 1000 --exemplar-selection herding "
		memoryoff_cmd=""
	else
		memoryon_cmd="--num-exemplars $memory_size --exemplar-selection herding "
		memoryoff_cmd="$memoryon_cmd"
	fi

	momlron_cmd="--momentum 0.9 --lr-factor 10 --lr-min 0.0000001 "
	momlroff_cmd=""

	for k in ${seeds[@]}; do
		seed_cmd="--seed ${k} "
        for fs in ${fseeds[@]}; do
            fseed_cmd="--fseed ${fs} "
            for lc in ${last_class[@]}; do
                last_class_cmd="--last-class $lc "
                for model in ${models[@]}; do
                    if [[ " ${mlmodels[*]} " =~ " ${model} " ]]; then
                        model_cmd="--ml-model ${model} "
                    else
                        model_cmd="--network ${model} "
                    fi
                    if [[ " ${hdrnets[*]} " =~ " ${model} " ]]; then
                        input_cmd=${hdrinput_cmd}
                    elif [[ " ${loadnets[*]} " =~ " ${model} " ]]; then
                        input_cmd=${loadinput_cmd}
                    else
                        input_cmd="${hdrinput_cmd} ${loadinput_cmd}"
                    fi
                    for b in ${nc_base[@]}; do
                        nc_base_cmd="--nc-first-task ${b} "

                        for i in ${nc_incr[@]}; do
                            nc_incr_cmd="--nc-incr-tasks ${i} "
                            if [[ $stop_task != 'none' ]]; then
                                s=$stop_task
                            elif [[ $i -ne 0 ]]; then
                                s=$(( (($max_classes - $b) / $i) + 1 ))
                            else
                                s=1
                            fi
                            stop_task_cmd="--stop-at-task ${s} "
                            num_tasks_cmd="--num-tasks ${s} "

                            expname_cmd="--exp-name ${exp_name}_base${b}_incr${i}_stop${s} "

                            for appr in ${approaches[@]}; do
                                approach_cmd="--approach ${appr} "
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
                                for sc in ${scale[@]}; do
                                    scale_cmd="--scale ${sc} "
                                    for sht in ${shots[@]}; do
                                        shots_cmd="--shots ${sht}"
                                        
                                        echo "${script_cmd}${expname_cmd}${respath_cmd}${dataset_cmd}${input_cmd}${nc_base_cmd}${nc_incr_cmd}${stop_task_cmd}${num_tasks_cmd}${config_cmd}${model_cmd}${approach_cmd}${memory_cmd}${momlr_cmd}${seed_cmd}${last_class_cmd}${out_feat_size_cmd}${fseed_cmd}${scale_cmd}${shots_cmd}" >> "${command_file}"
                                    done
                                done
                            done
                        done
                    done
				done
			done
		done
	done
else
	for failed_cmd in ${failed_cmds[@]}; do
		cat $failed_cmd >> $command_file
	done
	for g in $(seq 0 4); do
		sed -i "s|gpu $g|gpu $gpu_id|g" $command_file
	done
fi

# The delay is introduced in order to obtain a different timestamping for each experiment
sleep_command_file=${command_file}_sleep.log
gap=30  # in seconds
cnt=0
while read cmd_line; do
	sleep_t=$(( cnt % (gap * n_cpu) ))
	echo "sleep $sleep_t && $cmd_line" >> $sleep_command_file
	cnt=$(( cnt + gap ))
done <$command_file

xargs --arg-file $sleep_command_file --max-proc $n_cpu --replace --verbose /bin/sh -c "{}" 2> $exp_dir/logger.log
