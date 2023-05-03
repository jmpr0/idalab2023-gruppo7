from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name 8_tasks_herding_lopezcnn_2feats"
                        " --num-tasks 8"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 100"
                        " --last-layer-analysis"
                        " --save-models"
                        " --approach bic")


def test_bic_exemplars_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --num-exemplars 1000"
    args_line += " --exemplar-selection herding"
    run_main_and_assert(args_line)

# def test_bic_exemplars_lopezcnn_2feats17():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_bic_exemplars_lopez17rnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_bic_exemplars_lopez17cnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_bic_exemplars_lopez17cnnrnn():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
#     args_line += " --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# # def test_bic_exemplars():
# #     args_line = FAST_LOCAL_TEST_ARGS
# #     args_line += " --num-exemplars 200"
# #     run_main_and_assert(args_line)
# #
# #
# # def test_bic_exemplars_lambda():
# #     args_line = FAST_LOCAL_TEST_ARGS
# #     args_line += " --num-exemplars 200"
# #     args_line += " --lamb 1"
# #     run_main_and_assert(args_line)
# #
# #
# # def test_bic_exemplars_per_class():
# #     args_line = FAST_LOCAL_TEST_ARGS
# #     args_line += " --num-exemplars-per-class 20"
# #     run_main_and_assert(args_line)
# #
# #
# # def test_bic_with_warmup():
# #     args_line = FAST_LOCAL_TEST_ARGS
# #     args_line += " --warmup-nepochs 1"
# #     args_line += " --warmup-lr-factor 0.5"
# #     args_line += " --num-exemplars 200"
# #     run_main_and_assert(args_line)
