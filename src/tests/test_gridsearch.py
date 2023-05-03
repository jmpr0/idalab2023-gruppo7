from tests import *

FAST_LOCAL_TEST_ARGS = ("--exp-name local_test"
                        " --num-tasks 2"
                        " --seed 1"
                        " --batch-size 32"
                        " --nepochs 1"
                        " --gridsearch-tasks 3"
                        " --gridsearch-config gridsearch_config"
                        " --gridsearch-acc-drop-thr 0.2"
                        " --gridsearch-hparam-decay 0.5")


def test_gridsearch_finetuning():
    args_line = FAST_LOCAL_TEST_ARGS + MIMETIC_CONFIG
    args_line += " --approach finetuning --num-exemplars 200"
    run_main_and_assert(args_line)

# def test_gridsearch_freezing():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach freezing --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_joint():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach joint"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_lwf():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach lwf --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_icarl():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach icarl --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_eeil():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach eeil --nepochs-finetuning 1 --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_bic():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach bic --num-bias-epochs 1 --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_lucir():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach lucir --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_lwm():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach lwm --gradcam-layer conv2 --log-gradcam-samples 16 --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_ewc():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach ewc --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_mas():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach mas --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_pathint():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach path_integral --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_rwalk():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach r_walk --num-exemplars 200"
#     run_main_and_assert(args_line)
#
#
# def test_gridsearch_dmc():
#     args_line = FAST_LOCAL_TEST_ARGS + LOPEZ_CNN2_CONFIG
#     args_line += " --approach dmc"
#     args_line += " --aux-dataset mirage_generic --num-inputs 784"  # just to test the grid search fast
#     run_main_and_assert(args_line)
