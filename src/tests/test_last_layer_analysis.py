from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 2"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --lr-factor 10"
                        " --momentum 0.9"
                        " --lr-min 1e-7"
                        " --approach finetuning")


def test_last_layer_analysis():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --last-layer-analysis"
    run_main(args_line)
