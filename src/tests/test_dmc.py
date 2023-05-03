from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 3"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --approach dmc")

def test_dmc_lopezcnn_2feats17():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    run_main_and_assert(args_line)


def test_dmc_lopez17rnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
    run_main_and_assert(args_line)


def test_dmc_lopez17cnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    run_main_and_assert(args_line)


def test_dmc_lopez17cnnrnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
    run_main_and_assert(args_line)


def test_dmc_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + MIMETIC_CONFIG
    run_main_and_assert(args_line)

# def test_dmc():
#     run_main_and_assert(FAST_LOCAL_TEST_ARGS)
