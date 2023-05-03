from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name 8_tasks_herding_lopezcnn_krandom"
                        " --num-tasks 8"
                        " --seed 6"
                        " --batch-size 32"
                        " --nepochs 100"
                        " --last-layer-analysis"
                        " --save-models"
                        #" --lr-factor 10"
                        #" --momentum 0.9"
                        #" --lr-min 1e-7"
                        " --approach scratch")


def test_scratch_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN_CONFIG
    run_main_and_assert(args_line)

# def test_scratch_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_scratch_exemplars_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_scratch_lopez17rnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_scratch_lopez17cnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_scratch_lopez17cnnrnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_scratch():
#     args_line = FAST_LOCAL_TEST_ARGS
#     run_main_and_assert(args_line)
