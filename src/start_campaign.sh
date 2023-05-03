# WARNING: executed for seeds [5,9]
n_proc="./processors_15.dat"
seeds=(5 6 7 8 9 10)

for seed in ${seeds[@]}; do
	#campaign_name="incoherence_assessment"
	#for b in 2 4 8 16 32; do
	#	./start_incoherence_assessment_experiments.sh -e $campaign_name -c $n_proc -b $b -i 2 -s 2 -k $seed
	#	./start_incoherence_assessment_experiments.sh -e $campaign_name -c $n_proc -b $b -i 4 -s 2 -k $seed
	#	./start_incoherence_assessment_experiments.sh -e $campaign_name -c $n_proc -b $b -i 8 -s 2 -k $seed
	#done

	campaign_name="finetuning_momentum"
	./start_finetuning_momentum_experiments.sh -e $campaign_name -c $n_proc -b 20 -i 5 -s 2 -k $seed
done
