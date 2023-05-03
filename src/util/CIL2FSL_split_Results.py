import enum
from re import M
import pandas as pd
import numpy as np
import sys,os, glob, getopt
from pathlib import Path
import shutil
import json

fieldout={
"distance_metric": "euclidean", 
 "svm_C_reg": 0.1, 
 "svm_max_iters": 15, 
 "adaptation_steps": 1, 
 "adaptation_lr": 0.1, 
 "distance": "l2", 
 #"train_ways": 8, 
 #"train_shots": 25, 
 "train_queries": 25, 
 #"test_ways": 8, 
 #"test_shots": 25, 
 "test_queries": 50, 
 "lr": 0.0001, 
 "scheduler_step": 20, 
 "scheduler_decay": 1.0, 
 "call_forward": False, 
 "data_parallel": False, 
 "embedding_shape": [1, 24, 24], 
 "inner_channels": 8, 
 "apply_fce": False, 
 "lstm_layers": 1, 
 "unrolling_steps": 2, 
 "logger": True, 
 "checkpoint_callback": True, 
 "default_root_dir": "../new_results/mirage_generic_protonet_ctr_8_sht_25_epc_200_pat_20_net_lopez17cnn_scl_1-0", 
 "gradient_clip_val": 0, 
 "process_position": 0, 
 "num_nodes": 1, 
 #"num_processes": 1, 
 "gpus": 1, 
 "auto_select_gpus": False, 
 "log_gpu_memory": None, 
 "progress_bar_refresh_rate": 1, 
 "overfit_batches": 0.0, 
 "track_grad_norm": -1, 
 "check_val_every_n_epoch": 1, 
 "fast_dev_run": False, 
 "accumulate_grad_batches": 1, 
 #"max_epochs": 200, 
 "min_epochs": 1, 
 "max_steps": None, 
 "min_steps": None, 
 "limit_train_batches": 1.0, 
 "limit_val_batches": 1.0, 
 "limit_test_batches": 1.0, 
 "val_check_interval": 1.0, 
 "flush_logs_every_n_steps": 100, 
 "log_every_n_steps": 50, 
 "accelerator": None, 
 "sync_batchnorm": False, 
 "precision": 32, 
 "weights_summary": "top", 
 "weights_save_path": None, 
 "num_sanity_val_steps": 2, 
 "truncated_bptt_steps": None, 
 "resume_from_checkpoint": None, 
 "profiler": None, "benchmark": False, 
 "deterministic": False, 
 "reload_dataloaders_every_epoch": False, 
 "auto_lr_find": False, 
 "replace_sampler_ddp": True, 
 "terminate_on_nan": False, 
 "auto_scale_batch_size": False, 
 "prepare_data_per_node": True, 
 "amp_backend": "native", 
 "amp_level": "O2", 
 "distributed_backend": None, 
 "automatic_optimization": True, 
 "approach": "protonet", 
 "seed": 0, 
 "cleanup": True, 
 #"dataset": "mirage_generic", 
 "acc_grad_batches": 10, 
 "meta_batches_per_epoch": 10, 
 "num_tasks": -1, 

 "scale": 1.0, 
 "monitor": "valid_accuracy", 
 "min_delta": 0.001, 
 "patience": 20, 
 "mode": "max", 
 "device": "cuda"}

argsCIL2FSL={
    #Key:toFSL Value:fromCIL
    'nc_train':'nc_first_task',
    'nc_test': 'nc_incr_tasks',
    'train_ways':'nc_incr_tasks',
    'test_ways':'nc_incr_tasks',
    'test_shots':'shots',
    'max_epochs': 'nepochs',
    'num_processes':'num_workers',
    'train_shots':'shots',
    'dataset':'datasets',
    'acc_grad_batches':'batch_size',
    'default_root_dir': 'results_path',
}    
    
{     
    #'base_model_path', 
    #'batch_size', 
    #'clipping', 
    #'datasets', 
    #'debug', 
    #'eval_on_train', 
    #'exemplar_selection', 
    #'exp_name', 
    #'first_lr', 
    #'first_lr_factor', 
    #'first_lr_min', 
    # 'first_momentum', 
    # 'fix_bn', 
    # 'freeze_after', 
    # 'git_sha1', 
    # 'gpu', 
    # 'gridsearch_tasks', 
    # 'is_networking', 
    # 'last_class', 
    # 'last_layer_analysis', 
    # 'load_base_model', 
    # 'log', 
    #'lr_factor', 
    #'lr_min', 
    #'lr_patience', 
    #'modspec_nepochs', 
    #'momentum', 
    #'multi_softmax', 
    #'nc_first_task', 
    #'nc_incr_tasks', 
    #'nepochs':"max_epochs", 
    #'no_cudnn_deterministic', 
    #'num_exemplars', 
    #'num_exemplars_per_class', 
    #'num_workers':"num_processes", 
    #'pin_memory', 
    #'pre_allocated_output', 
    #'pretrained', 
    #'results_path', 
    #'save_models', 
    #'shots':'test_shots', 
    #'stop_at_task', 
    #'timestamp_to_recover', 
    # 'use_valid_only', 
    # 'validation', 
    # 'warmup_lr_factor', 
    # 'warmup_nepochs', 
    # 'weight_decay'
}

def fix_args(res_fn):
    ts=os.path.basename(res_fn).split('_')[-1].split('.')[0].split('-')[1]
    ts= ts.split('-')[1] if len(ts.split('-'))==2 else ts
    argsf=os.path.dirname(res_fn).replace('/backup','').replace('/results','')+'/args-%s.txt'%ts
    with open(argsf,'r') as f:
        args = json.load(f)
    for k,v in argsCIL2FSL.items():
        args[k]=args[v]
    with open(argsf,'w') as f:
        json.dump(args,f)
def main(argv):
    
    results_root_path=None
    force=False
    try:
        opts, args = getopt.getopt(argv, "hr:f", "[input=]")
    except getopt.GetoptError:
        print('hierarchical_lopez_session.py -s <source>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(' ')
            sys.exit()
        if opt in ("-r", "--results"):
            results_root_path=arg
            if not os.path.exists(results_root_path):
                print('Results Root Path doesn\'t exists')
                exit()
        if opt in ("-f", "--force"):
            force=True
    results_list=glob.glob(results_root_path+'/**/*outputs_targets_features_*.parquet', recursive=True)
    if len(results_list)==0:
        print('Results file not found')
        exit()
    
    task_id={
        0: 'train',
        1: 'test'
    }
    results_list=[res_fn for res_fn in results_list if 'metrics' not in res_fn and 'train' not in res_fn and 'test' not in res_fn and 'update' not in res_fn and 'val' not in res_fn]
    assert len(results_list)!=0
    if not force:
        results_list=[res_fn for res_fn in results_list if 'backup' not in res_fn]
        
    for res_fn in results_list:
        tsparq=os.path.basename(res_fn).split('_')[-1]
        backup_dir=os.path.dirname(res_fn)+'/backup/'
        Path(backup_dir).mkdir(parents=True,exist_ok=True)
        bname=os.path.basename(res_fn)
        
        try:
            df=pd.read_parquet(res_fn)
        except:
            print('W: error in reading<%s>'%res_fn)
            corrupted_dir=os.path.dirname(res_fn)+'/corrupted/'
            Path(corrupted_dir).mkdir(parents=True,exist_ok=True)
            shutil.move(res_fn,corrupted_dir+bname)
            continue
        for tid,task in task_id.items():
            if task=='test':
                try:
                    dfx=df.iloc[tid,:]
                except:
                    print('%s [%s] not found'%(task,tid))
                    print(res_fn)
                    print(df)
                for i,k in enumerate(['update','test']):
                    dft={}
                    for c in df.columns:
                        dft[c]=[[dfx[c][i]]]
                    dft=pd.DataFrame(dft, columns=df.columns)
                    # if k=='test':
                    #     print('--->',dft) 
                    res_fn_out=res_fn.replace(tsparq,'best_%s_%s'%(k,tsparq))

                    if 'backup' not in res_fn:
                        dft.to_parquet(res_fn_out) 
                    else:
                        res_fn_out=res_fn_out.replace('backup/','')
                        dft.to_parquet(res_fn_out)
            else:    
                dft=df.iloc[[tid],:]
                res_fn_out=res_fn.replace(tsparq,'best_%s_%s'%(task,tsparq))
                
                if 'backup' not in res_fn:
                    dft.to_parquet(res_fn_out) 
                else:
                    res_fn_out=res_fn_out.replace('backup/','')
                    dft.to_parquet(res_fn_out)
        if 'backup' not in res_fn:
            shutil.move(res_fn,backup_dir+bname)
        fix_args(res_fn)
if __name__=='__main__':
    main(sys.argv[1:])

