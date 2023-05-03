from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name 8_tasks_herding_lopezcnn_2feats"
                        " --num-tasks 8"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 100"
                        " --last-layer-analysis"
                        " --save-models"
                        " --approach freezing")


def test_freezing_exemplars_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    run_main_and_assert(args_line)

# def test_freezing_exemplars_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_freezing_exemplars_lopez17rnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_freezing_exemplars_lopez17cnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_freezing_exemplars_lopez17cnnrnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_freezing_exemplars_aceto19mimetic():
#     args_line = FAST_LOCAL_TEST_ARGS + MIMETIC_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)

# def test_freezing_without_exemplars():
#     run_main_and_assert(FAST_LOCAL_TEST_ARGS)
# 
# 
# def test_freezing_with_exemplars():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
# 
# 
# def test_freezing_with_warmup():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --warmup-nepochs 1"
#     args_line += " --warmup-lr-factor 0.5"
#     run_main_and_assert(args_line)
