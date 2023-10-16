from helper.replayer import Replayer
import torch
import torch.nn as nn
from generative_model.linear_nets import fc_layer,fc_layer_split

from helper.utils import relative_to_abs


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]): # dim_list means layers neuron number. eg: [64,128,356,128,64]
        layers.append(nn.Linear(dim_in, dim_out))            # define a layer needs input num and output num , so 连续两个为一组的生成layer
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)   # * 解包，从list 中，把元素释放出来，作为sequential的输入

class PoolHiddenNet(nn.Module):   # should be social encoder part.(FC+pooling)
    '''Pooling module as proposed in social-gan'''
    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
            activation='relu', batch_norm=True, dropout=0.0
    ):
        super(PoolHiddenNet, self).__init__()  # 调用父类 nn.Module 的构造函数，确保在子类的构造函数中也执行了父类的初始化操作

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, self.mlp_dim, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)  # input x,y 2 dimension
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def repeat(self, tensor, num_reps):    # 这个方法用于将输入的张量在第一维上进行重复，以构建不同维度的重复数据。主要用于在计算相对位置时进行数据复制
        """
        Inputs:
        -tensor: 2D tensor of any shape        #我理解为 dim time, dim features(x,y)  写成2D(t,f)
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2    #  ((x0,y0);(x0,y0);(x1,y1);(x1,y1))  0,1 means time, repeat for calculate relative
        """
        col_len = tensor.size(1)  # when 2D ， size 1 is col。 when 3 D size 1 is row
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)   # unsqueeze get 3D (t,1,f), repeat get 3D (t,num,f), 由于从右开始看 row 对应num， col 对应 f， 所以会说是repeat row
        tensor = tensor.view(-1, col_len) # size(num*t, f)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):       ## ?  h_states might not comes from Class Predictor, which forward() doesn't provide h_states, only provides position output
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)    encoder的final state 一个batch一个batch的输入   ??cause we use nn.LSTMCell I guess num_layers = 1 . right
        - seq_start_end: A list of tuples which delimit sequences within batch    #在批处理中分隔序列的元组列表# 这个batch/pedestrains 输入的处于同一个场景的的peds index (start_id,end_id) 构成一个tuple。这些tuple把总共的batch区分成几个scene 。 切片操作是基于 seq_start_end 中的 start 和 end 索引进行的,它指定了当前轨迹序列在展平后的2D张量中的起始和结束位置。
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):   # [(start, end),(start, end),..]   是3D （seq, ped_id(batch), pos(x,y)） 转 2D 后的 tensor的对应start， end index。
            start = start.item()
            end = end.item()
            num_ped = end - start    # ped pedestrain               可以得到当前轨迹序列中的行人数量， 所以一个循环处理的是这个batch中的所有trajectory/ 因为用的cumsum 方法， 按每组的人数累加的，（start，end），包含start 不包含 end。
            if num_ped > 1:
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]  # (num_layers(LSTM的layer，我们取1) * batch, h_dim)  不同的start 和end 引起不同的loop， 从而代表不同的trajectory
                curr_end_pos = end_pos[start:end]                       
                # Repeat -> H1, H2, H1, H2                          增加行， 有多少行人，复制多少次。 整体一起copy
                curr_hidden_1 = curr_hidden.repeat(num_ped, 1) 
                # Repeat position -> P1, P2, P1, P2
                curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
                # Repeat position -> P1, P1, P2, P2
                curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)    #                    每个batch的行人数量不一定一样
                curr_rel_pos = curr_end_pos_1 - curr_end_pos_2         # relative position!
                curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
                mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)  #          这俩都是2D了， 扩展列=增加一个特征，复制后的hidden state。 为什么a对a或b或c的相对位置分别和 a或b或c的hidden state 拼起来再送入感知机MLP： 我觉得是模拟一个人观察自己和他人的距离，以及他人的运动轨迹（信息压缩在他人的hidden中），才做出自己的判断。
                curr_pool_h = self.mlp_pre_pool(mlp_h_input)                         # 2d（复制后的行数num_ped*num_ped，新列数 是embedding 输出特征的维度数 ）
                curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]    # dim = 1  reduce dim1. 这个max就是 max pool 的由来。 [0]是只返回max的值而不返回index
            else:                                
                curr_hidden = h_states.view(-1, self.h_dim)[start:end]
                curr_pool_h = curr_hidden
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class AutoEncoder(Replayer):
    '''Class for variational auto-encoder (VAE) models.'''

    def __init__(
            self,
            obs_len=12,
            pred_len=8,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=124,
            traj_lstm_output_size=2,
            dropout=0,
            z_dim=200,
            embedding_dim=32,
            mlp_dim=256,
            bottleneck_dim=32,
            activation='relu',
            batch_norm=True
    ):
        # Set configurations
        super().__init__()
        self.label = "VAE"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size
        self.z_dim = z_dim

        # Weights of different components of the loss function
        self.lamda_rcl = 1.
        self.lamda_vl = 1.
        self.lamda_pl = 0.

        self.average = "average" # --> makes that [reconL] and [variatL] are both divided by number of iput-pixels

        # pooling configurations
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.bottleneck_dim = bottleneck_dim



        ###---------SPECIFY MODEL--------###

        ##>---Encoder (=q[z|x])---<##
        # -flatten traj to 2D-tensor

        # -hidden state
        self.traj_lstm_model_encoder = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        # self.traj_lstm_model_encoder2= nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        self.fcE = fc_layer(traj_lstm_hidden_size + bottleneck_dim, 128, batch_norm=None)

        # -to z
        self.toZ = fc_layer_split(128, z_dim, nl_mean='none', nl_logvar='none')


        ##>---Decoder (=p[x|z])---<##
        # -from z
        self.fromZ = fc_layer(z_dim, 128, batch_norm=None)
        # -fully connected hidden layers
        self.fcD = fc_layer(128, traj_lstm_hidden_size, batch_norm=None)

        # -hidden state
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size)
        # -to traj
        self.pred_hidden2pos = nn.Linear(self.traj_lstm_output_size, 2)

        # -pooling   #todo
        self.pool_net = PoolHiddenNet(
            embedding_dim=self.embedding_dim,
            h_dim=traj_lstm_hidden_size,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )


    @property
    def name(self):
        return "{}".format("Generator --> VAE")


    ##---- FORWARD FUNCTIONS ----##

    # initial observe traj lstm hidden states
    def init_obs_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    # initial prediction traj lstm hidden states
    def init_pred_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    # Pass input through feed-forward connections, to get [hE], [z_mean] and [z_logvar]
    def encode(self, x):
        # extract final hidden features (forward-pass)
        hE = self.fcE(x)
        # get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)         # 输出维度可以定义，可以不是1维度的。
        return z_mean, z_logvar, hE

    # Perform "reparametrization trick" to make these stochastic variables differentiable
    def reparameterize(self, mu, logvar):    # 改成 多维度时，这里应该要改
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()      # .normal_() GPT 说是单维度的。
        return eps.mul(std).add_(mu)

    def decode(self, z, seq_start_end, obs_traj_pos=None):
        hD = self.fromZ(z)   # 从z中展开，是个MLP
        hidden_features = self.fcD(hD)  # embedding 得到相同的特征维度。
        pred_lstm_h_t = hidden_features
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        pred_traj_pos = []

        output = obs_traj_pos[0]    # 输入的是-1. end position， decode时候就是 start position了
        pred_lstm_h_t = self.pool_net(hidden_features, seq_start_end, output)  # output 是obs的路径的 end position # 再pool一次。在那片论文里可以再decode每一次过lstm都pool一次，毕竟附近的人的距离会变化。
        pred_traj_pos += [output]    # output是(batch, 2)，也是一个时间步的结果。
        for i in range(self.obs_len-1):        #  # ？？lack a cancatenate with hidden state again as input to LSTM like that paper？ 有个问题， 那篇论文再decode部分 采用的是 pool后的结果和上一步的hidden state 拼接起来进LSTM的。！
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                output, (pred_lstm_h_t, pred_lstm_c_t)  # todo whether use teach force, input_t --> output
            )
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]
        outputs = torch.stack(pred_traj_pos)
        return outputs



    # Pass latent variable activations through feedback connections, to generator reconstructed image
    # def decode(self, z):
    #     hD = self.fromZ(z)


    def forward(self, obs_traj_pos, seq_start_end):   # obs_traj_pos （seq_len, batch/ped, input_size(x,y)=2）  seq_start_end 指定这个batch的哪些ped是一组的，seq_start_end内有几个tuple也制定了一共有几组scene。
        batch = obs_traj_pos.shape[1] #todo define the batch
        traj_lstm_h_t, traj_lstm_c_t = self.init_obs_traj_lstm(batch)
        # traj_lstm_h_t_2, traj_lstm_c_t_2 =self.init_obs_traj_lstm(batch)
        # pred_lstm_h_t, pred_lstm_c_t = self.init_pred_traj_lstm(batch)
        traj_lstm_hidden_states = []
        traj_lstm_hidden_states_2 = []
        pred_lstm_hidden_states = []

        # Encoder1, calculate the past traj hidden states, similar with embedding
        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(
                obs_traj_pos[: self.obs_len].size(0), dim=0
            )  # （1，batch，input_size） 每个时间步逐步输入LSTM
        ):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model_encoder(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )  # （batch，input_size）
            traj_lstm_hidden_states += [traj_lstm_h_t]

        # Encoder2, calculate the future traj hidden states, similar with embedding
        # for i, input_t in enumerate(
        #     obs_traj_pos[-self.pred_len :].chunk(
        #         obs_traj_pos[-self.pred_len :].size(0), dim=0
        #     )
        # ):
        #     traj_lstm_h_t_2, traj_lstm_c_t_2 = self.traj_lstm_model_encoder2(
        #         input_t.squeeze(0), (traj_lstm_h_t_2, traj_lstm_c_t_2)
        #     )
        #     traj_lstm_hidden_states_2 += [traj_lstm_h_t_2]

        # complete_input = torch.cat(
        #     (traj_lstm_hidden_states[-1], traj_lstm_hidden_states_2[-1]), dim=0
        # )    #
        
        # encode (forward), reparameterize and decode (backward)
        final_encoder_h  = traj_lstm_hidden_states[-1]
        # social pooling (Reference:https://github.com/agrimgupta92/sgan)
        end_pos = obs_traj_pos[-1, :, :]       # 最后一个可观测时间步的所有batch的位置信息。 接下来就要拆开batch得到多个scene进入 pool 了，之前LSTM 只考虑了每个行人的轨迹。batch/ped id不同的都不会互相影响 
        pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)    # 压缩的信息意义是： 行人a考虑其他人对a的距离 以及 其他人对应的hidden state（历史轨迹行为） 后，经过MLP (考虑)，最被激活的代表 最需要考虑的特征。（可能反应的是b在身边很近但同方向，c从对面来很快且靠近...）
        # Construct input hidden states for decoder
        vae_input = torch.cat([final_encoder_h, pool_h], dim=1)            # 由于pool 考虑的是和他人的信息，忘记了自己的路径（类似忘记了自己的目的地和所在地），所以要cat
        # vae_input = final_encoder_h
        mu, logvar, hE = self.encode(vae_input)      
        z = self.reparameterize(mu, logvar)   # 自带sample
        traj_recon = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos)
        return (traj_recon, mu, logvar, z) 



    ##------- SAMPLE FUNCTIONS -------##

    def sample(self, obs_traj_rel, obs_traj, replay_seq_start_end): # 所谓的replay_seq_start_end 就是用新的task的dataset取的数据格式，来创建replay数据！ 即additional info中的number
        '''Generate [size] samples from the model. Output is tensor (not "requiring grad"), on same device as <self>'''

        # set model to eval()-mode
        mode = self.training   
        self.eval()
        obs_traj_rel = obs_traj_rel
        replay_seq_start_end = replay_seq_start_end
        size = obs_traj_rel.shape[1]

        # sample z 
        z = torch.randn(size, self.z_dim).to(self._device())  # 修改成多维高斯，这里采样可能也要变？

        # decode z into traj x
        with torch.no_grad():
            traj_rel = self.decode(z, replay_seq_start_end, obs_traj_pos=obs_traj_rel)

        # relative to absolute
        traj = relative_to_abs(traj_rel, obs_traj[0]) # obs_traj 即 additional info中的 intial position

        # set model back to its initial mode
        self.train(mode=mode)
        replay_traj = [traj, traj_rel, replay_seq_start_end]
        # returen samples as [batch_size]x[traj_size] tensor
        return replay_traj

    ##-------- LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, mode=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]         <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]   (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]   <bool>, if True, loss is average over all frames; otherwise it is summed

        OUTPUT: - [reconL]    <1D-tensor> of length [batch_size]
        '''
        # x = x.permute(1,0,2)
        # x_recon = x_recon.permute(1,0,2)
        # batch_size = x.size(0)
        seq_len, batch, size = x.size()
        # reconL = F.binary_cross_entropy(input=x_recon.reshape(batch_size, -1), target=x.reshape(batch_size, -1),
        #                                 reduction='none')
        # reconL = torch.mean(reconL, dim=1) if mode else torch.sum(reconL, dim=1)

        reconL = (x.permute(1,0,2) - x_recon.permute(1,0,2)) ** 2
        if mode == "sum":
            return torch.sum(reconL)
        elif mode == "average":
            return torch.sum(reconL) / (batch*seq_len*size)   # 这是对所有元素求平均，没有忽略batch
        elif mode == "raw":
            return reconL.sum(dim=2).sum(dim=1)


    def calculate_variat_loss(self, mu, logvar):
        '''Calculate reconstruction loss for each element in the batch. 

        INPUT:  - [mu]      <2D-tensor> by encoder predicted mean for [z]
                - [logvar]  <2D-tensor> by encoder predicted logvar for [z]   # 2d (batch,dimension)

        OUTPUT: - [variatL] <1D-tensor> of length [batch_size]
        '''

        # --> calculate analytically
        # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
        variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)    #可以用于多维高斯，因为每个mu，logvar元素独立。这里也是对dim1 求和，样本维度上求和。

        return variatL

    def loss_function(self, recon_x, x, y_hat= None, y_target=None, scores=None, mu=None, logvar=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:   - [recon_x]     <4D-tensor> reconstructed traj in same shape as [x]
                 - [x]           <4D-tensor> original traj
                 - [y_hat]       <2D-tensor> predicted traj
                 - [y_target]    <2D-tensor> future traj
                 - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
                 - [logvar]      None or <2D-tensor> with estimated log(SD^2) of [z]

        SETTING: - [self.average] <bool>, if True, both [reconL] and [variatL] are divided by number of input elements

        OUTPUT:  - [reconL]      reconstruction loss indicating how well [x] and [x_recon] match
                 - [variatL]     variational (KL-divergence) loss "indicating how normally distributed [z] is"
                 - [predL]       prediction loss indicating how well targets [y] are predicted
                 - [distilL]     knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                    match the target "logits" ([scores])
        '''

        ###---Reconstruction loss---###
        reconL = self.calculate_recon_loss(x=x, x_recon=recon_x, mode=self.average)  # -> possibly average over traj
        reconL = torch.mean(reconL)                                                     # -> average over batch   # 其实没必要，上一步对所有元素平均了

        ###--- Variational loss ----###
        if logvar is not None:
            variatL = self.calculate_variat_loss(mu=mu, logvar=logvar)
            variatL = torch.mean(variatL)                               # -> average over batch
            if self.average:
                pass         # todo
        else:
            variatL = torch.tensor(0., device=self._device())

        '''
        ###----Prediction loss----###
        if y_target is not None:
            predL = F.cross_entropy(y_hat, y_target, reduction='mean')  #-> average over batch
        else:
            predL = torch.tensor(0., device=self._device())
        '''
        # Return a tuple of the calculated losses
        return reconL, variatL

    ##------- TRAINING FUNCTIONS -------##

    def train_a_batch(self, x_rel, y_rel, seq_start_end, x_=None, y_=None, seq_start_end_=None, rnt=0.5):
        '''Train model for one batch ([x],[y]),possibly supplemented with replayed data ([x_],[y_])

        [x]          <tensor> batch of past trajectory (could be None, in which case only 'replayed' data is used)
        [y]          <tensor> batch of corresponding future trajectory
        [x_]         None or (<list> of) <tensor> batch of replayed past trajectory
        [y_]         None or (<list> of) <tensor> batch of corresponding future trajectory
        [rnt]        <number> in [0,1], relative importance of new task
        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x_rel is not None:

            # Run the model
            recon_batch, mu, logvar, z = self(x_rel, seq_start_end)

            # If needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in current task
            # if active_classes is not None:
            #     pass              #

            # Calculate all losses    # 因为是VAE，是重建，所以没有yhat  ##YZ internal replay 情况下，x是 hidden_in （进入SGR的hidden，），x_recon 是hidden_out(调用full的forward，可以得到).
            reconL, variatL = self.loss_function(recon_x=recon_batch, x=x_rel, y_hat=None, y_target=None, mu=mu, logvar=logvar)

            # Weigh losses as requested
            # loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL + self.lamda_pl*predL
            loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL

            # Calculate training-precision  #


        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:
            #YZ# the BI code 说TASKIL 任务 y_ 是list， x_ 也是list。既然下文判断了x_是不是list，那前面也应该同样的方法对y_ 判断。
            TaskIL = (type(y_)==list) 
            if not TaskIL:
                y_ = [y_]
            ## 这样和encoder里面的写法一致了。 这样这里因该也是1了  。 debug验证也是1.
            #     
            n_replays = len(y_) if (y_ is not None) else 1
            # n_replays = 1

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            reconL_r = [None]*n_replays
            variatL_r = [None]*n_replays
            predL_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task)
            if (not type(x_)==list):
                x_temp_ = x_
                recon_batch, mu, logvar, z = self(x_temp_, seq_start_end_)
            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list):
                    x_temp_ = x_[replay_id]
                    recon_batch, mu, logvar, z = self(x_temp_)

                # Calculate all losses
                reconL_r[replay_id], variatL_r[replay_id] = self.loss_function(
                    recon_x=recon_batch, x=x_temp_, mu=mu, logvar=logvar
                )

                # Weigh losses as requested
                loss_replay[replay_id] = self.lamda_rcl*reconL_r[replay_id] + self.lamda_vl*variatL_r[replay_id]
                '''
                if self.replay_target=="hard":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                elif self.replay_target=="soft":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                '''

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        loss_total = loss_replay if (x_rel is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)



        # Backpropagate errors
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'reconL': reconL.item() if x_rel is not None else 0,
            'variatL': variatL.item() if x_rel is not None else 0,
            # 'predL': predL.item() if x is not None else 0,
            'reconL_r': sum(reconL_r).item()/n_replays if x_ is not None else 0,
            'variatL_r': sum(variatL_r).item()/n_replays if x_ is not None else 0,
            # 'predL_r': sum(predL_r).item()/n_replays if x_ is not None else 0,
        }
