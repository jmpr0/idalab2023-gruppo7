from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 2"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --lr-factor 10"
                        " --momentum 0.9"
                        " --lr-min 1e-7")


def test_finetuning_without_warmup():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)


def test_finetuning_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --approach finetuning"
    args_line += " --warmup-nepochs 1"
    args_line += " --warmup-lr-factor 0.5"
    run_main_and_assert(args_line)
