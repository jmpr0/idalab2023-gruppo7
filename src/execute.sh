processors=$1
command_file=$2
executed_command_file=$(echo $command_file | sed 's|\.log|_executed\.log|g')

fshift() { local v n=$'\n';read -r v < <(
    sed -e $'1{w/dev/stdout\n;d}' -i~ "$1")
    printf ${2+-v} $2 "%s${n[${2+2}]}" "$v"
}

fpush() { sed -e "\$a$2" -i~ "$1"; }

curr_cpu=
while [ -z $curr_cpu ]; do
	curr_cpu=$(flock $processors -c "sed -e $'1{w/dev/stdout\n;d}' -i~ $processors")
	if [ -z $curr_cpu ]; then
		# select a random number of seconds to wait (60-180 seconds)
		t=$(( 60 + $RANDOM % 120 + 1 ))
		echo "[$$] $command_file waiting for $t"
		taskset 1 sleep $t
	fi
done

exp_cmd=$(flock $command_file -c "./get_line.sh $command_file")

echo "[$$] exec $exp_cmd"
taskset $curr_cpu sh -c "$exp_cmd"

flock $processors -c "echo $curr_cpu >> $processors"
flock $executed_command_file -c "echo $exp_cmd >> $executed_command_file" 
