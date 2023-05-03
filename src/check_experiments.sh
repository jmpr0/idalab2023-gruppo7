exp_name=$1
results_path=$2

if [ -z "$exp_name" ]; then
	read -p "Please, set the experiment name [local_test]: " exp_name
	if [ -z "$exp_name" ]; then
		echo "Using default experiment name local_test."
		exp_name="local_test"
	fi
fi
while [ -z "$results_path" ]; do
	read -p "Please, set the path to the results: " results_path
done

search=($results_path/*$exp_name/)
args_files=$(find "${search[@]}" -iname "args-*")

campaign_ts=$(date +"%s")
err_command_file="err_commands_${campaign_ts}.log"

script_cmd="python3 main_incremental.py"

for args_file in $args_files; do
	stdout_file=$(echo $args_file | sed 's|args|stdout|g')
	stderr_file=$(echo $args_file | sed 's|args|stderr|g')
	stdout_done=$(grep "Done!" $stdout_file | wc -l | awk '{print $1}')
	stderr_size=$(wc -c $stderr_file | awk '{print $1}')
	if [[ $stdout_done == 0 ]] && [[ $stderr_size != 0 ]]; then
		echo "Experiment $args_file failed."
		args=$(echo -e "import json\nargs=json.loads(open('$args_file').read())\nprint(' '.join(['--%s %s' % (k.replace('_', '-'), args[k] if not isinstance(args[k], list) else ' '.join('%s' % v for v in args[k])) if not isinstance(args[k], bool) else '--%s' % k.replace('_', '-') for k in args if (args[k] or not isinstance(args[k], (bool, list))) and k != 'is_networking' and not args[k] is None]))" | python3)
		echo "$script_cmd $args" >> $err_command_file
	fi
done
