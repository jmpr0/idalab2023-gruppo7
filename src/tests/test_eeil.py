from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 2"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --approach eeil")


def test_eeil_exemplars_lopezcnn_2feats17():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --num-exemplars 200"
    args_line += " --nepochs-finetuning 1"
    run_main_and_assert(args_line)


def test_eeil_exemplars_lopez17rnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_RNN_CONFIG
    args_line += " --num-exemplars 200"
    args_line += " --nepochs-finetuning 1"
    run_main_and_assert(args_line)


def test_eeil_exemplars_lopez17cnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
    args_line += " --num-exemplars 200"
    args_line += " --nepochs-finetuning 1"
    run_main_and_assert(args_line)


def test_eeil_exemplars_lopez17cnnrnn():
    args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2RNN_CONFIG
    args_line += " --num-exemplars 200"
    args_line += " --nepochs-finetuning 1"
    run_main_and_assert(args_line)


def test_eeil_exemplars_aceto19mimetic():
    args_line = FAST_LOCAL_TEST_ARGS + MIMETIC_CONFIG
    args_line += " --num-exemplars 200"
    args_line += " --nepochs-finetuning 1"
    run_main_and_assert(args_line)

# def test_eeil_exemplars_with_noise_grad():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --num-exemplars 200"
#     args_line += " --nepochs-finetuning 1"
#     args_line += " --noise-grad"
#     run_main_and_assert(args_line)
#
#
# def test_eeil_exemplars():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --num-exemplars 200"
#     args_line += " --nepochs-finetuning 1"
#     run_main_and_assert(args_line)
#
#
# def test_eeil_with_warmup():
#     args_line = FAST_LOCAL_TEST_ARGS
#     args_line += " --warmup-nepochs 1"
#     args_line += " --warmup-lr-factor 0.5"
#     args_line += " --num-exemplars 200"
#     args_line += " --nepochs-finetuning 1"
#     run_main_and_assert(args_line)
