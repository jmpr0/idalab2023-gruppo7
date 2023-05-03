from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name 8_tasks_herding_lopezcnn_2feats"
                        " --num-tasks 8"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 100"
                        " --last-layer-analysis"
                        " --save-models"
                        #" --lr-factor 10"
                        #" --momentum 0.9"
                        #" --lr-min 1e-7"
                        " --approach joint")


def test_joint_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    run_main_and_assert(args_line)

# def test_joint_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_joint_exemplars_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_joint_lopez17rnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_joint_lopez17cnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_joint_lopez17cnnrnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
#     run_main_and_assert(args_line)
#
#
# def test_joint():
#     args_line = FAST_LOCAL_TEST_ARGS
#     run_main_and_assert(args_line)
