from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 5"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --stop-at-task 2")


def test_finetuning_stop_at_task():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)
