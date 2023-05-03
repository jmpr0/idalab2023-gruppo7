import os
import glob
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    l.sort(key=alphanum_key)
    
def delete_files(files, checkpoint_epochs):
    sort_nicely(files)
    files.pop()
    for file in files:
        if sum([f'ep{checkpoint_epoch}.' in file for checkpoint_epoch in checkpoint_epochs]):
            continue
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f'File does not exist -> {file}')
    
def clean_files(path):
    """
    Delete queries and support files except the ones from the last epoch
    """
    folders = ['train_data/', 'val_data/', 'test_data/']
    # Retrieving the number of epoch for each checkpoint
    checkpoint_epochs = [v.split('=')[-1].split('.')[0] for v in glob.glob(f'{path}/checkpoints/*')]
    
    for folder in folders:
        query_files = glob.glob(f'{path}/{folder}queries_ep*')
        support_files = glob.glob(f'{path}/{folder}supports_ep*')
        loss_files = glob.glob(f'{path}/{folder}losses_ep*')
        
        for files in [query_files, support_files, loss_files]:
            if len(files) <= 2:
                continue
            delete_files(files, checkpoint_epochs)

if __name__ == '__main__':
    PATH='' 
    clean_files(PATH)