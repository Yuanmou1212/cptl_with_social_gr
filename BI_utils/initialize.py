# initialize use pretrained parameters.
# initialize weights and bias
# 记得回到args去加俩参数
from torch import nn
import os
import torch
def initialize_model(model,args):
    # re-initialzie all params
    model.apply(weight_reset)  # iterably go through layers.
    # initialize weight
    if args.init_weight ==True:
        weight_init(model,strategy="xavier_normal")
    # initialize bias
    if args.init_bias == True :
        bias_init(model,strategy="constant")
    # load pre_trained params (we use so far best SGR model's parameters)
    if args.hidden ==True:
        file_dir = os.path.join(os.path.dirname(__file__),"..",'model_params','BEST_MODEL')
        file_name = os.path.join(file_dir,'BEST.path')
        check_point = torch.load(file_name)
        # LSTM only
        weight_ih=check_point['traj_lstm_model.weight_ih']#.state_dict() traj_lstm_model.weight_ih,... is print out
        weight_hh=check_point['traj_lstm_model.weight_hh']
        bias_ih = check_point['traj_lstm_model.bias_ih']
        bias_hh = check_point['traj_lstm_model.bias_hh']
        model.traj_lstm_model.weight_ih.data = weight_ih
        model.traj_lstm_model.weight_hh.data = weight_hh
        model.traj_lstm_model.bias_ih.data = bias_ih
        model.traj_lstm_model.bias_hh.data = bias_hh

    return model

## weight initialize , only influene FC,layrs. no LSTM
def weight_init(model, strategy="xavier_normal",std=0.01):
    # xavior_noraml, xaviro_uniform, normal 3 mode
    if hasattr(model,"list_init_layers"):
        module_list=model.list_init_layers()
        params = [p for m in module_list for p in m.parameters()]
    else:
        params = [p for p in model.parameters()]

    for p in params:
        if p.dim()>=2:
            if strategy=="xavier_normal":
                nn.init.xavier_normal_(p)
            elif strategy == "xavier_uniform":
                nn.init.xavier_uniform_(p)
            elif strategy == "normal":
                nn.init.normal_(p,std=std)
            else:
                raise ValueError("Invalid weight-initialization strategy {}".format(strategy))

## bias initialize
def bias_init(model,strategy="constant",value=0.01):
    # zero,constant,positive,any 4 mode
    if hasattr(model,"list_init_layers"):
        module_list=model.list_init_layers()
        params = [p for m in module_list for p in m.parameters()]
    else:
        params = [p for p in model.parameters()]

    for p in params:
        if p.dim()>=2:
            if strategy == "zero":
                nn.init.constant_(p, val=0)
            elif strategy == "constant":
                nn.init.constant_(p, val=value)
            elif strategy == "positive":
                nn.init.uniform_(p, a=0, b=value)
            elif strategy == "any":
                nn.init.uniform_(p, a=-value, b=value)
            else:
                raise ValueError("Invalid bias-initialization strategy {}".format(strategy))

## rest linear
def weight_reset(m): # m:module
    '''Reinitializes parameters of [m] according to default initialization scheme.'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()