err_command_file=$1
processors=$2

while [ ! -f "$err_command_file" ]; do
	read -p "Please, set the err_command_file path: " err_command_file
done
while [ ! -f "$processors" ]; do
	read -p "Please, set the processors path: " processors
done

n_exp=$(cat $err_command_file | wc -l)
for exp_id in $(seq $n_exp); do
	./execute.sh $processors $err_command_file &
	taskset 1 sleep 30
done
