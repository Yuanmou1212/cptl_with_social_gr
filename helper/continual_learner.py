import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a predictor'''

    def __init__(self):   #attributes and hyperparameters used for different continual learning techniques.
        super().__init__()

        # XdG:
        self.mask_dict = None  # -> <dict> with task-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # -SI:
        self.si_c = 0  # -> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = 0.1  # -> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # -EWC:
        self.ewc_lambda = 0  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = True  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0  # -> keeps track of number of quadratic loss terms (for "offline EWC")

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda



    @abc.abstractmethod
    def forward(self, x):
        pass



    #------------- "Synaptic Intelligence Synapses"-specifc functions -------------#

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters(): # n name , p parameter
            if p.requires_grad:     # select trainable items from iterator  只有可训练的参数才拿来用
                n = n.replace('.', '__')   #字符串中的点换成下划线 /避免 和python语法冲突

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))    # {}是占位符，值由n 来填充。。 这里的attribute的名字是SI_prev_task, 所以是找到previous的参数。至于怎么存的，继续看
                p_current = p.detach().clone()   # 分离 克隆             # train line 85 将送来训练的模型的初始参数（训练之前）放进了'{}_SI_prev_task'， 所以这里能找到！
                p_change = p_current - p_prev                           # 而update_omega 在每个epoch训练结束后才调用，所以p current确实不同
                omega_add = W[n]/(p_change**2 + epsilon)  # epsilon 保证分母不为0   ##? how to get W
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))           # try 找这个omega，没找到就在这初始化
                except AttributeError:   #如果没有 对应的attribute，则进入下面的  
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)  ## Q：compared with omega, prev have no try..except.. block, how to make sure there is prev at very beginning? 在每个epoch结束时调用
                self.register_buffer('{}_SI_omega'.format(n), omega_new)      ## SI_prev 在train.py 一开始初始化， SI_omega 在这try 这里初始化。


    def surrogate_loss(self):  # encoder.py , train_a_batch will call it.
        '''Calculate SI's surrogate loss.'''  ## need omega (regularization strength) and  change in parameter values
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self, '{}_SI_prev_task'.format(n))  # SI_prev in train.py have intialization, so we can search it  
                    omega = getattr(self, '{}_SI_omega'.format(n))            #  只要update_omega 先， 就有omega
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
