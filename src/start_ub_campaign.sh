# WARNING: executed for seeds [5,9]
n_proc="./processors_15.dat"
seeds=(0 1 2 3 4 5 6 7 8 9 10)
bs=(2 4 8 16)
is=(1 2 3)
s=2

campaign_name="upperbound"

for b in ${bs[@]}; do
for i in ${is[@]}; do
for seed in ${seeds[@]}; do
	./start_experiments.sh -e $campaign_name -a "scratch" -r "../hf-project/results_UB" -c $n_proc -b $b -i $i -s $s -k $seed
done
done
done
