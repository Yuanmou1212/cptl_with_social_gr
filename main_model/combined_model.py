# model_RTF.encoder_lstm.parameters() , need freeze encoder_lstm,so we should separate it here!
# not just lstm, also include FC at begining! 

# this model should include sample function, in train_BI, previous_generator is = previous_model now.

# 类似的实例化调用方法。
# model = Predictor(
#                     obs_len=args.obs_len,
#                     pred_len=args.pred_len,
#                     traj_lstm_input_size=args.traj_lstm_input_size,
#                     traj_lstm_hidden_size=args.traj_lstm_hidden_size,
#                     traj_lstm_output_size=args.traj_lstm_output_size
#                 ).to(device)
# 要加入hidden=args.hidden  默认set False， 调用记得输入


#### same dependency
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


class Model_RTF(Replayer):
    '''Class for variational auto-encoder (VAE) models.'''

    def __init__(
            self,
            obs_len=12,
            pred_len=8,
            traj_lstm_input_size=2,
            traj_lstm_hidden_size=124,
            traj_lstm_output_size=2,
            hidden=False,
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
        self.lamda_pl = 1.

        self.lamda_gen = 1.
        self.lamda_main = 1.

        self.average = "average" # --> makes that [reconL] and [variatL] are both divided by number of iput-pixels

        # pooling configurations
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.bottleneck_dim = bottleneck_dim

        # BI-methods configurations
        self.hidden = hidden # internal replay
       


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
        # -pooling   #todo
        self.pool_net = PoolHiddenNet(
            embedding_dim=self.embedding_dim,
            h_dim=traj_lstm_hidden_size,
            mlp_dim=mlp_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout )
        
        ##>--- Predictor (from hidden state to traj)---<##
        # -hidden state
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size)
        # -to traj
        self.pred_hidden2pos = nn.Linear(self.traj_lstm_output_size, 2)

        
       


    @property
    def name(self):
        return "{}".format("Generator --> VAE")

    def list_init_layers(self):
        # return list of layers can be intilize by xavier and related methods(except lstm)
        list=[]
        list+=self.fcE.list_init_layers()
        list+=self.toZ.list_init_layers()
        list+=self.fromZ.list_init_layers()
        list+=self.fcD.list_init_layers()
        # list+=self.pool_net.list_init_layers()
        # list+=self.pred_hidden2pos.list_init_layers()
        return list

    # initial encoder traj lstm hidden states
    def init_encoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),     # intialize hidden_state and cell_state
        )


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

    def decode(self, z, seq_start_end, obs_traj_pos=None): # decode 中的 最后一个LSTM 替换成identity， 然后 写个跳过中途的VAE的 encode # YZ 
        hD = self.fromZ(z)   # 从z中展开，是个MLP
        hidden_features = self.fcD(hD)  # embedding 得到相同的特征维度。
        pred_lstm_h_t = hidden_features
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()
        pred_traj_pos = []

        output = obs_traj_pos[0]    # 输入的是-1. end position， decode时候就是 start position了 # 反转了，因为这是VAE 的步骤，目的是为了重构，而不是预测，所以这里是找到obs_traj 的起点！ 所以是0
        pred_lstm_h_t = self.pool_net(hidden_features, seq_start_end, output)  # output 是obs的路径的 end position # 再pool一次。在那片论文里可以再decode每一次过lstm都pool一次，毕竟附近的人的距离会变化。
        decode_h = pred_lstm_h_t  # before LSTM.
        pred_traj_pos += [output]  ## output是(batch, 2)，也是一个时间步的结果。
        for i in range(self.obs_len-1):        #  # ？？lack a cancatenate with hidden state again as input to LSTM like that paper？ 有个问题， 那篇论文再decode部分 采用的是 pool后的结果和上一步的hidden state 拼接起来进LSTM的。！
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(
                output, (pred_lstm_h_t, pred_lstm_c_t)  # todo whether use teach force, input_t --> output
            )
            output = self.pred_hidden2pos(pred_lstm_h_t)
            pred_traj_pos += [output]
        outputs = torch.stack(pred_traj_pos)
        return outputs,decode_h



    # Pass latent variable activations through feedback connections, to generator reconstructed image
    # def decode(self, z):
    #     hD = self.fromZ(z)

    # def predict(self,input,not_hidden=True,U_info=None):
    #     # input can be x if not_hidden = True,  input can be hidden if not_hidden= false
    
    #     if  not_hidden==True: # input is obs_traj_pos, from X
    #         batch = input.shape[1]
    #         traj_lstm_h_t, traj_lstm_c_t = self.init_obs_traj_lstm(batch)
    #         pred_traj_pos = []
    #         traj_lstm_hidden_states = []

    #         for i, input_t in enumerate(
    #             input[: self.obs_len].chunk(   ## ：Doc shows chunk() input should be chunks rather than size of one chunk. chunks=size0 of input, so we have same number(dim0 of input) of chunks 
    #                 input[: self.obs_len].size(0), dim=0  # 就是把单个时间步的tensor[1,...]给提了出来（input_t)用来循环。 其实取切片也可以做到
    #             )      # (1，batch, input_size)
    #         ):
    #             #print(input_t.shape)   # input shape ([1, 64, 2]) 后面的batch出现过 epoch([1, 72, 2]) torch.Size([1, 107, 2]) torch.Size([1, 143, 2]) 等等  （time，batch （ped），position ）
    #             traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model_encoder(
    #                 input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)  # dim 0 of input_t is 1, use squeeze to remove this dim  # 初始化的值都是二维的 squeeze掉的只能是时间，按照LSTM 的输入要求来看。留下的是【batch，input_size】 
    #             )
    #             traj_lstm_hidden_states += [traj_lstm_h_t]  # same like .append()
    #         final_encoder_h  = traj_lstm_hidden_states[-1]
    #         pred_lstm_h_t = final_encoder_h
    #         pred_lstm_c_t=torch.zeros_like(final_encoder_h).cuda()
    #         end_pos = input[-1, :, :]

    #         for i in range(self.pred_len):
    #             pred_lstm_h_t,pred_lstm_c_t = self.pred_lstm_model(
    #                 end_pos,(pred_lstm_h_t,pred_lstm_c_t)
    #             )
    #             output = self.pred_hidden2pos(pred_lstm_h_t)
    #             pred_traj_pos += [output]
    #         traj_pred = torch.stack(pred_traj_pos)
    #         return traj_pred
        
    #     else: # input from hidden, after encoder lstm.(actually from replay)
    #         end_pos,_= U_info
    #         final_encoder_h = input
    #         pred_traj_pos =[]
    #         pred_lstm_c_t=torch.zeros_like(final_encoder_h).cuda()
    #         pred_lstm_h_t = final_encoder_h

    #         for i in range(self.pred_len):
    #             pred_lstm_h_t,pred_lstm_c_t = self.pred_lstm_model(
    #                 end_pos,(pred_lstm_h_t,pred_lstm_c_t)
    #             )
    #             output = self.pred_hidden2pos(pred_lstm_h_t)
    #             pred_traj_pos += [output]
    #         traj_pred = torch.stack(pred_traj_pos)
    #         return traj_pred


    # def forward(self, obs_traj_pos, seq_start_end,full=False,input_not_hidden=True,U_info=None):   # obs_traj_pos （seq_len, batch/ped, input_size(x,y)=2）  seq_start_end 指定这个batch的哪些ped是一组的，seq_start_end内有几个tuple也制定了一共有几组scene。
    #     ## if full==false, only predict! full==true, go through Social GR.
    #     if full==True:
    #         batch = obs_traj_pos.shape[1] #todo define the batch
    #         traj_lstm_h_t, traj_lstm_c_t = self.init_obs_traj_lstm(batch)
    #         # traj_lstm_h_t_2, traj_lstm_c_t_2 =self.init_obs_traj_lstm(batch)
    #         # pred_lstm_h_t, pred_lstm_c_t = self.init_pred_traj_lstm(batch)
    #         traj_lstm_hidden_states = []
    #         traj_lstm_hidden_states_2 = []
    #         pred_lstm_hidden_states = []

    #         # Encoder1, calculate the past traj hidden states, similar with embedding
    #         for i, input_t in enumerate(
    #             obs_traj_pos[: self.obs_len].chunk(
    #                 obs_traj_pos[: self.obs_len].size(0), dim=0
    #             )  # （1，batch，input_size） 每个时间步逐步输入LSTM
    #         ):
    #             traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model_encoder(
    #                 input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
    #             )  # （batch，input_size）  # shared encode part #YZ
    #             traj_lstm_hidden_states += [traj_lstm_h_t]

            
    #         # encode (forward), reparameterize and decode (backward)
    #         final_encoder_h  = traj_lstm_hidden_states[-1]
    #         # social pooling (Reference:https://github.com/agrimgupta92/sgan)
    #         end_pos = obs_traj_pos[-1, :, :]       # 最后一个可观测时间步的所有batch的位置信息。 接下来就要拆开batch得到多个scene进入 pool 了，之前LSTM 只考虑了每个行人的轨迹。batch/ped id不同的都不会互相影响 
            
    #         ## Generator part
            
    #         pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)    # 压缩的信息意义是： 行人a考虑其他人对a的距离 以及 其他人对应的hidden state（历史轨迹行为） 后，经过MLP (考虑)，最被激活的代表 最需要考虑的特征。（可能反应的是b在身边很近但同方向，c从对面来很快且靠近...）
    #         # Construct input hidden states for decoder
    #         vae_input = torch.cat([final_encoder_h, pool_h], dim=1)            # 由于pool 考虑的是和他人的信息，忘记了自己的路径（类似忘记了自己的目的地和所在地），所以要cat
    #         # vae_input = final_encoder_h
    #         mu, logvar, hE = self.encode(vae_input)       # VAE encode
    #         z = self.reparameterize(mu, logvar)   # 自带sample
    #         #traj_recon = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos) 
    #         # implement internal replay. if hidden=true, output = hidden state before LSTm, else, output= outputs position
    #         traj_recon,decode_h = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos)          
    #         return (traj_recon,mu,logvar,z,decode_h) #(traj_recon, mu, logvar, z) 
        
    #     ## Predict part
    #     else: # only run main model= predict part.

    #         if input_not_hidden == True:
    #             traj_pred = self.predict(obs_traj_pos,not_hidden=True,U_info=U_info) # 只进行最初的main_model！ default is True and none
    #         else:
    #             traj_pred = self.predict(obs_traj_pos,not_hidden=False,U_info=U_info)
    #         return traj_pred


    def forward(self,obs_traj_pos, seq_start_end,input_not_hidden=True,U_info=None,obs_traj_record=None):  # 精简版，一代二，同时输出x_recon 和 y_pred
            if input_not_hidden == True: # input is x.  
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
                    )  # （batch，input_size）  # shared encode part #YZ
                    traj_lstm_hidden_states += [traj_lstm_h_t]

                # encode (forward), reparameterize and decode (backward)
                final_encoder_h  = traj_lstm_hidden_states[-1]
                # social pooling (Reference:https://github.com/agrimgupta92/sgan)
                end_pos = obs_traj_pos[-1, :, :]       # 最后一个可观测时间步的所有batch的位置信息。 接下来就要拆开batch得到多个scene进入 pool 了，之前LSTM 只考虑了每个行人的轨迹。batch/ped id不同的都不会互相影响 
            
            elif input_not_hidden == False: # input is replay hidden state(when use internal replay)
                final_encoder_h = obs_traj_pos  # when replay is input, obs_traj_pos is actually a hidden state
                end_pos,_ = U_info
                obs_traj_pos = obs_traj_record # when replay, trajectory here is used to decode, cause decode.py just want to get start_pos through it!

            ## Generator part (vae generator)
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)    # 压缩的信息意义是： 行人a考虑其他人对a的距离 以及 其他人对应的hidden state（历史轨迹行为） 后，经过MLP (考虑)，最被激活的代表 最需要考虑的特征。（可能反应的是b在身边很近但同方向，c从对面来很快且靠近...）
            # Construct input hidden states for decoder
            vae_input = torch.cat([final_encoder_h, pool_h], dim=1)            # 由于pool 考虑的是和他人的信息，忘记了自己的路径（类似忘记了自己的目的地和所在地），所以要cat
            # vae_input = final_encoder_h
            mu, logvar, hE = self.encode(vae_input)       # VAE encode
            z = self.reparameterize(mu, logvar)   # 自带sample
            #traj_recon = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos) 
            # implement internal replay. if hidden=true, output = hidden state before LSTm, else, output= outputs position
            traj_recon,decode_h = self.decode(z, seq_start_end, obs_traj_pos=obs_traj_pos)          
            
            # Predict Part (main model)
            pred_lstm_c_t=torch.zeros_like(final_encoder_h).cuda()
            pred_lstm_h_t = final_encoder_h
            pred_traj_pos =[]
            for i in range(self.pred_len):
                pred_lstm_h_t,pred_lstm_c_t = self.pred_lstm_model(
                    end_pos,(pred_lstm_h_t,pred_lstm_c_t)
                )
                output = self.pred_hidden2pos(pred_lstm_h_t)
                pred_traj_pos += [output]
            traj_pred = torch.stack(pred_traj_pos)

            return (traj_pred,traj_recon,mu,logvar,z,decode_h) #(traj_recon, mu, logvar, z) 
        




    ##------  Suppliment function -------##
    # # function to only run SGR part from hidden state(h_in) to hidden state(h_out). especially for internal replay.
    # def forward_hidden_SGR (self,h_in,seq_start_end_,obs_traj_pos,U_info):  # obs_traj_pos is just for get the start point, I didn't change the name only want to reduce parameters needed to change.
    #     end_pos,_ = U_info
    #     pool_h = self.pool_net(h_in, seq_start_end_, end_pos) 
    #     vae_input = torch.cat([h_in, pool_h], dim=1) 
    #     mu, logvar, hE = self.encode(vae_input) 
    #     z = self.reparameterize(mu, logvar)
    #     traj_recon,h_out = self.decode(z, seq_start_end_, obs_traj_pos=obs_traj_pos)
    #     # predict result:
    #     pred_traj=self.predict(input=h_in,not_hidden=False,U_info=U_info)
    #     return (traj_recon,mu,logvar,h_out,pred_traj)

    ##------- SAMPLE FUNCTIONS -------##
    ## 需要sample既能replay batch data，又能replay batch hidden state+ U info (end_pos，seq_start_end)
    def sample(self, obs_traj_rel, obs_traj, replay_seq_start_end): # 所谓的replay_seq_start_end 就是用新的task的dataset取的数据格式，来创建replay数据！ 即additional info中的number
        '''Generate [size] samples from the model. Output is tensor (not "requiring grad"), on same device as <self>'''

        # set model to eval()-mode
        mode = self.training   
        self.eval()
        obs_traj_rel = obs_traj_rel
        replay_seq_start_end = replay_seq_start_end
        size = obs_traj_rel.shape[1]

        # sample z 
        z = torch.randn(size, self.z_dim).to(self._device())  # if修改成多维高斯，这里采样可能也要变？
        
        if self.hidden == False:
            # decode z into traj x
            with torch.no_grad():
                traj_rel,_ = self.decode(z, replay_seq_start_end, obs_traj_pos=obs_traj_rel)
            # relative to absolute
            traj = relative_to_abs(traj_rel, obs_traj[0]) # obs_traj 即 additional info中的 intial position 
            replay_traj = [traj, traj_rel, replay_seq_start_end]
            # returen samples as [batch_size]x[traj_size] tensor
            # set model back to its initial mode
            self.train(mode=mode)
            return replay_traj
        else:
            # decode z into hidden state
            with torch.no_grad():
                _,decode_h = self.decode(z, replay_seq_start_end, obs_traj_pos=obs_traj_rel)
            
            end_pos = obs_traj[-1]
            U_info = (end_pos,replay_seq_start_end)
                # 所以调用时，少了个参数，得对应
            return decode_h,U_info
    
        

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
    # sample后得到 应该得到一个hidden，送进SGR 即forward full 中，得到 hidden_recon， 这俩直接求loss，来训练SGR part/ SGR 的训练在哪儿？
    # 这里是copy的 vae的train a batch。应该看看main的train a batch
    def train_a_batch(self, x_rel, y_rel, seq_start_end, U_info=None,x_=None, y_=None, seq_start_end_=None, rnt=0.5,obs_traj_record=None,loss_mask=None): # more input U_info
        '''Train model for one batch ([x],[y]),possibly supplemented with replayed data ([x_],[y_])

        [x]          <tensor> batch of past trajectory (could be None, in which case only 'replayed' data is used)
        [y]          <tensor> batch of corresponding future trajectory
        [x_]         None or (<list> of) <tensor> batch of replayed past trajectory ## if use internal replay, x_rel_ is hidden state.
        [y_]         None or (<list> of) <tensor> batch of corresponding future trajectory
        [rnt]        <number> in [0,1], relative importance of new task
        [U_info]       (end_pos,replay_seq_start_end )
        obs_traj_record    obs_traj for current dataset, is called for replay, decode get start position through it.
        '''

        # Set model to training-mode
        self.train()
        # If frozen. set that part eval()
        if self.hidden == True:
            self.traj_lstm_model_encoder.eval()

        # Reset optimizer
        self.optimizer.zero_grad()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x_rel is not None:  # train main model. YZ

            ## Run the model 
            y_hat_rel,recon_traj, mu, logvar, z,decode_h = self.forward(x_rel, seq_start_end)  ## YZ  default input_not_hidden=True input is current data(must be x),run trhough forward function, get 2 pathes result.
            
            
            # If needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in current task
            # if active_classes is not None:
            #     pass              #

            ## internal replay or not
            # x go through encoder lstm get x_h
            
            if self.hidden == True: # replace x_rel with x_h
                batch = x_rel.shape[1]
                traj_lstm_h_t, traj_lstm_c_t = self.init_encoder_traj_lstm(batch)   # (batch,32) 
                for i, input_t in enumerate(
                    x_rel[: self.obs_len].chunk(   ## ：Doc shows chunk() input should be chunks rather than size of one chunk. chunks=size0 of input, so we have same number(dim0 of input) of chunks 
                    x_rel[: self.obs_len].size(0), dim=0  # 就是把单个时间步的tensor[1,...]给提了出来（input_t)用来循环。 其实取切片也可以做到
                ) ):

                    traj_lstm_h_t,_ = self.traj_lstm_model_encoder(input_t.squeeze(0),(traj_lstm_h_t, traj_lstm_c_t)) if self.hidden else x_rel
                x_rel = traj_lstm_h_t.unsqueeze(0) # replace x_rel with last x_h, and change size form(64,32)into(1,64,32) just to suit loss calculation
            # else: x_rel = x_rel

            # recon_x is recon x or recon hidden
            recon_x  = decode_h.unsqueeze(0) if self.hidden else recon_traj

            # Calculate generator losses    # 因为是VAE，是重建，所以没有yhat
            reconL, variatL = self.loss_function(recon_x=recon_x, x=x_rel, y_hat=None, y_target=None, mu=mu, logvar=logvar)
                # due to the LSTM input x is (seq,batch/ped, H_in),  and output hidden state is (seq,batch,H_out). Therefore the changed x and recon_x are suitable for loss_function

            # Calculate predictor loss
            from helper.utils import l2_loss
            predL = None if y_rel is None else l2_loss(y_hat_rel, y_rel, mode="average")
            
            # Weigh losses as requested
            # loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL + self.lamda_pl*predL
            loss_cur = self.lamda_gen*(self.lamda_rcl*reconL + self.lamda_vl*variatL) + self.lamda_main*self.lamda_pl*predL
            

            pred_traj = loss_cur
            # Calculate training-precision  #


        ##--(2)-- REPLAYED DATA --##
        if x_ is not None:  # x_ is get by sample. when internal replay, x_ is hidden state.
            if not (type(y_)==list): # check y_ is a list of tensor or 
                y_=[y_]
            n_replays = len(y_) if (y_ is not None) else 1
            # n_replays = 1

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays
            reconL_r = [None]*n_replays
            variatL_r = [None]*n_replays
            predL_r = [None]*n_replays
            pred_traj_r = [None]*n_replays

            # Run model (if [x_] is not a list with separate replay per task)
            if (not type(x_)==list):
                x_temp_ = x_  # when internal replay, x_ is hidden state.
                #recon_batch, mu, logvar, z = self.forward(x_temp_, seq_start_end_)
                if self.hidden == True:
                    pred_y,recon_traj,mu,logvar,z,recon_hidden=self.forward(obs_traj_pos=x_temp_,seq_start_end=seq_start_end_,input_not_hidden=False,obs_traj_record=obs_traj_record,U_info=U_info)
                    recon=recon_hidden.unsqueeze(0) # is hidden # 不确定是不是这么改！！ replay'data似乎没有原始路径。即输入的data是x的情况。
                    x_temp_ = x_temp_.unsqueeze(0) # is hidden
                else: # x_temp_  is original input 
                    pred_y,recon_traj, mu, logvar, z,recon_hidden=self.forward(obs_traj_pos=x_temp_,seq_start_end=seq_start_end_) # default is input_not_hidden=True
                    recon = recon_traj # is x
            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_)==list): # debug test, will not go into this.
                    x_temp_ = x_[replay_id]
                    if self.hidden == True:
                        pred_y,recon_traj,mu,logvar,z,recon_hidden = self.forward(obs_traj_pos=x_temp_,seq_start_end=seq_start_end_,input_not_hidden=False,obs_traj_record=obs_traj_record,U_info=U_info)
                        # note: recon_traj seq is 8 , pred_traj seq is 12, different!
                        recon=recon_hidden.unsqueeze(0)  
                        x_temp_ = x_temp_.unsqueeze(0)
                    else:
                        pred_y,recon_traj, mu, logvar, z,recon_hidden=self.forward(obs_traj_pos=x_temp_,seq_start_end=seq_start_end_) # ,input_not_hidden=True
                        recon = recon_traj # is x
                # Calculate generator losses
                reconL_r[replay_id], variatL_r[replay_id] = self.loss_function(
                    recon_x=recon, x=x_temp_, mu=mu, logvar=logvar
                )

                # Calculate predict loss
                from helper.utils import l2_loss
                predL_r[replay_id] = None if y_ is None else l2_loss(pred_y, y_[replay_id], mode="average")

                # Weigh losses as requested
                loss_replay[replay_id] = self.lamda_gen*(self.lamda_rcl*reconL_r[replay_id] + self.lamda_vl*variatL_r[replay_id])+self.lamda_main*self.lamda_pl*predL_r[replay_id]
                pred_traj_r[replay_id] = loss_replay[replay_id]
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
            'loss_current':loss_cur.item() if x_rel is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x_rel is not None) else 0,
            'reconL': reconL.item() if x_rel is not None else 0,
            'variatL': variatL.item() if x_rel is not None else 0,
            'predL': predL.item() if x_rel is not None else 0,
            'reconL_r': sum(reconL_r).item()/n_replays if x_ is not None else 0,
            'variatL_r': sum(variatL_r).item()/n_replays if x_ is not None else 0,
            'predL_r': sum(predL_r).item()/n_replays if x_ is not None else 0,
            'pred_traj': pred_traj.item() if pred_traj is not None else 0,
            'pred_traj_r': sum(pred_traj_r).item()/n_replays if (x_ is not None and pred_traj_r[0] is not None) else 0,
            
            }
