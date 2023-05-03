import torch
import torch.nn.functional as F

import pandas as pd

from torchsummary import summary
from torch import nn
from abc import abstractmethod

class Yang21CNN(nn.Module):
    def __init__(self, input_size=20, output_size=200, densesize=512, initialize_weights=True, **kwargs):
        super().__init__()

        self._metrics = {}
        self.input_size = input_size
        self.output_size = output_size

        self.c1= nn.Sequential(            
            nn.Conv1d(1, 128, kernel_size=(3,), stride=(1,), padding=(1,)),          
            nn.ReLU(),
            #torch.nn.BatchNorm1d(128, eps=1e-07, momentum=0, affine=False),
            nn.MaxPool1d(kernel_size=2, stride=2)         
        ) 
        
        self.c2= nn.Sequential(            
            nn.Conv1d(128, 96, kernel_size=(3,), stride=(1,), padding=(1,)),                            # N,C_in,L_in         
            nn.ReLU(),
            #torch.nn.BatchNorm1d(96, eps=1e-07, momentum=0, affine=False),
            nn.MaxPool1d(kernel_size=4, stride=2)         
        )     
        
        self.c3= nn.Sequential(            
            nn.Conv1d(96, 32, kernel_size=(3,), stride=(1,), padding=(1,)),
            nn.ReLU(),
            #torch.nn.BatchNorm1d(32, eps=1e-07, momentum=0, affine=False),  
            nn.Dropout(p=0.2),    
        )       
        
        self.dense = nn.Sequential(
            nn.Linear(
                int(32 * int(self.input_size / 4 - 1)), 
                int(densesize*2)
            ), 
            nn.ReLU(),
            nn.Linear(int(densesize*2), densesize),  
            nn.ReLU(),
                   
            #torch.nn.BatchNorm1d(densesize, eps=1e-07, momentum=0, affine=False),  
        )             
        
#        self.out = nn.Sequential(
#            nn.Linear(densesize, self.output_size),                  
#            nn.LogSoftmax(dim=1)
#        )
        self.fc = nn.Linear(densesize, self.output_size)
        self.head_var = 'fc'
        
        if initialize_weights:
            self.initialize_weights()


    def forward(self, x):
        x = self.extract_features(x) 
        y = self.fc(x)
        #y = nn.functional.log_softmax(x, dim=1)
        return y

    def extract_features(self, x):
        x = self.c1(x)
        x = self.c2(x)     
        x = self.c3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        return x


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def initialize_weights(self):
        for m in self.modules():
        
            if isinstance(m, nn.Conv1d):
                #print(m)
                nn.init.xavier_uniform_(m.weight, gain=1.0)     
                #  kaiming_normal_  , mode='fan_out', nonlinearity='relu' 0.8603445685038796
                #  xavier_uniform_    0.8603445685038796
                #nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm1d) and m.affine:
                #print(m)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                #print(m)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def summary(self, device:str='cuda'):
        if device is not None:
            device = device.split(':', 1)[0]
        return summary(self, (1, int(self.input_size)), device=device)

    def track_metric(self, metric_name, metric_value):
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(metric_value)
