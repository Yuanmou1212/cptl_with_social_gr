import torch
from torch.nn import functional as F
import torch.nn as nn
import copy


from helper.replayer import Replayer
from helper.continual_learner import ContinualLearner
from helper.utils import l2_loss

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()  # * 解包操作，tuple（2，3） 经 * =》 2,3
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()  # .sub_()   .mul_()  原地操作，动这个tensor， 减法和乘法
    raise ValueError('Unrecognized noise type "%Ss"' % noise_type)


class Predictor(ContinualLearner, Replayer): # nn module 是 continual learner 的父类，所以这里不用写了
    '''Model for predicting trajectory, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    # reference GAN code, generator part, encoder & decoder (LSTM)
    def __init__(
            self,
            obs_len,
            pred_len,
            traj_lstm_input_size,
            traj_lstm_hidden_size,
            traj_lstm_output_size,
            dropout=0,
            noise_dim=(8,),
            noise_type="gaussian",
    ):
        super().__init__()
        self.label = "lstm"
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_lstm_input_size = traj_lstm_input_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_output_size = traj_lstm_output_size

        self.noise_dim = noise_dim
        self.noise_type = noise_type


        #--------------------------MAIN SPECIFY MODEL------------------------#

        #-------Encoder-------# 
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size) # LSTMCell is single layer LSTM, and we call it's instance lot times, so we use the same LSTMCell at different time step, not in different deep!

        #-------Decoder------#                                                          # output have same size like input, so two LSTMcell model use same input_size
        self.pred_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_output_size) ## ? why not use(hidden_size,output_size) like Unet encoder-deocer----由于decoder的输入其实是 linear之后获得的pos，和轨迹中的pos一样都是x，y俩维度的。
        self.pred_hidden2pos =nn.Linear(self.traj_lstm_output_size, 2) # 2 means x,y position value

    # initial encoder traj lstm hidden states
    def init_encoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )
    # initial decoder traj lstm hidden states
    def init_decoder_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
            torch.randn(batch, self.traj_lstm_output_size).cuda(),
        )

    # add noise before decoder
    def add_noise(self, _input):
        noise_shape = (_input.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)
        decoder_h = torch.cat([_input, z_decoder], dim=1)
        return decoder_h


    @property
    def name(self):
        return "{}".format("lstm")

    def forward(self, obs_traj_pos, seq_start_end):
        batch = obs_traj_pos.shape[1] #todo define the batch
        traj_lstm_h_t, traj_lstm_c_t = self.init_encoder_traj_lstm(batch)
        # pred_lstm_h_t, pred_lstm_c_t = self.init_decoder_traj_lstm(batch)
        pred_traj_pos = []
        traj_lstm_hidden_states = []
        pred_lstm_hidden_states = []

        # encoder, calculate the hidden states

        for i, input_t in enumerate(
            obs_traj_pos[: self.obs_len].chunk(   ## ：Doc shows chunk() input should be chunks rather than size of one chunk. chunks=size0 of input, so we have same number(dim0 of input) of chunks 
                obs_traj_pos[: self.obs_len].size(0), dim=0  # 就是把单个时间步的tensor[1,...]给提了出来（input_t)用来循环。 其实取切片也可以做到
            )
        ):
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)  # dim 0 of input_t is 1, use squeeze to remove this dim
            )
            traj_lstm_hidden_states += [traj_lstm_h_t]  # same like .append()


        output = obs_traj_pos[self.obs_len-1]   ## index start from 0. so last observed data's index is obs_len-1 输入的轨迹中的最后一个值。（也就是decoder第一步输入的值）
        pred_lstm_h_t_before_noise = traj_lstm_hidden_states[-1]
        # pred_lstm_h_t = self.add_noise(pred_lstm_h_t_before_noise)
        pred_lstm_h_t = pred_lstm_h_t_before_noise
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t).cuda()  # clear c , long term memory

        for i in range(self.pred_len):
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(   ##  pred_lstm_h_t 是encoder的最后一个hidden结果。
                output, (pred_lstm_h_t, pred_lstm_c_t)        # 循环中的output（也就是LSTM中的X）是pos，在循环中依赖linear从hidden state中得到的pos
            )
            output = self.pred_hidden2pos(pred_lstm_h_t)      # output have same size like input, so two LSTMcell model use same input_size 
            pred_traj_pos += [output]
        outputs = torch.stack(pred_traj_pos)

        return outputs

    def train_a_batch(self, x_rel, y_rel, seq_start_end, x_rel_=None, y_rel_=None, seq_start_end_=None, loss_mask=None, active_classes=None, rnt=0.5):
        '''
        Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_], [y_]).

        [x]       <tensor> batch of past trajectory (could be None, in which case only 'replayed' data is used)
        [y]       <tensor> batch of corresponding future trajectory
        [x_]      None or (<list> of) <tensor> batch of replayed past trajectory
        [y_]      None or (<list> of) <tensor> batch of corresponding "replayed"  future trajectory

        '''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        #--(1)-- REPLAYED DATA---#

        if x_rel_ is not None:
            y_ = [y_rel_]
            n_replays = len(y_) if (y_ is not None) else None     # 我理解成 list里面有很多个tensor（每个tensor是一个batch的数据），n代表有多数个tensor

            # Prepare lists to store losses for each replay
            loss_replay = [None]*n_replays    # 将 None 重复 n_replays 次来创建列表
            pred_traj_r = [None]*n_replays
            distill_r = [None]*n_replays

            # Loop to evaluate predictions on replay according to each previous task
            y_hat_all = self(x_rel_, seq_start_end_)  # self means current instance. this way it == self.foward() function, will get predict results.

            for replay_id in range(n_replays):
                # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classed not in replayed task
                # y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]
                y_hat = y_hat_all   # 预测值 pred

                # Calculate losses
                if (y_rel_ is not None) and (y_[replay_id] is not None):
                    # pred_traj_r[replay_id] = F.cross_entropy(y_hat.permute(1,0,2), y_[replay_id].permute(1,0,2), reduction='mean')
                    pred_traj_r[replay_id] = l2_loss(y_hat, y_[replay_id], mode="average")

                # Weigh losses
                loss_replay[replay_id] = pred_traj_r[replay_id]

        # Calculate total replay loss
        loss_replay = None if (x_rel_ is None) else sum(loss_replay) / n_replays

        #--(2)-- CURRENT DATA --#

        if x_rel is not None:
            # Run model
            y_hat_rel = self(x_rel, seq_start_end)
            # relative to absolute
            # y_hat = relative_to_abs(y_hat_rel, )
            # Calculate prediction loss
            # pred_traj = None if y is None else F.cross_entropy(input=y_hat.permute(1,0,2), target=y.permute(1,0,2), reduction='mean')
            pred_traj = None if y_rel is None else l2_loss(y_hat_rel, y_rel, mode="average")    # 上面的输入是list of tensor，但这里输入是tensor
            # a = torch.numel(loss_mask.data)

            # Weigh losses
            loss_cur = pred_traj

        # Combine loss from current and replayed batch
        if x_rel_ is None:
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x_rel is None) else rnt*loss_cur+(1-rnt)*loss_replay  # rnt就是 replay 与current loss 占比权重。

        #--(3)-- ALLOCATION LOSSES --#

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()  # if it is the first task, no previous values, how to calculate it??   
        if self.si_c > 0:     ## when call it, default si_c is changed
            loss_total += self.si_c * surrogate_loss

        # Backpropagate errors (if not yet done)
        loss_total.backward()

        # Take optimization-step
        self.optimizer.step()

        # Returen the dictionary with different training-loss split in categories   ## I didn't find it showing "split in categories"
        return {
            'loss_total': loss_total.item(),
            'loss_current':loss_cur.item() if x_rel is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x_rel is not None) else 0,
            'pred_traj': pred_traj.item() if pred_traj is not None else 0,
            'pred_traj_r': sum(pred_traj_r).item()/n_replays if (x_rel_ is not None and pred_traj_r[0] is not None) else 0,
            'si_loss': surrogate_loss.item(),
        }




