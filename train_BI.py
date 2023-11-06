import tqdm
import copy
import torch
import os
from helper import utils
from helper.continual_learner import ContinualLearner
from data.loader import data_loader, data_dset
from torch.autograd import Variable
import shutil
from torch import optim

def train(args, model, train_loader, optimizer, epoch, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,   # （seq_len（frame）, batch（ped）, input_size) 真正batch 的体现是 ped这一维度，多个scene（每个都有几个ped）的ped全拼在一起，所以batch 0 不等于 第一个scene的数据，可能batch = 【0，。。，10】 的切片才是第一个scene 
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)   # 创建了一个标量（值为0）的损失张量，并将其移到（.to）与pred_traj_gt相同的设备上
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]  # dim0 是 纳入考虑的人数， dim 1 是 起点到重点的seq 的index， 现在是gt的部分的index了。 batch是按人数dim0这一维度拼接的。没有permute过。

        ###################################################
        model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)  # 输入lstm的是相对位置值(个体自己的相对值，不是和他人。)
        pred_traj_fake_rel,_,_,_,_,_ = model(model_input, seq_start_end)    # seq_start_end =  [[start,end],[..],[..]]  #  output (pred_len, batch,2)
        l2_loss_rel = utils.l2_loss(
            pred_traj_fake_rel,
            model_input[-args.pred_len:],
            loss_mask,
            mode="average",
        )

        loss = l2_loss_rel  
        losses.update(loss.item(), obs_traj.shape[1])    # loss是 tensor[1] 所以需要item 转化成纯数。 # 这个update是自定义函数，losses是utils.AverageMeter的instance，存loss用的
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)   # losses.avg 整个batch 的loss ， 在update函数里面除以了 batch size， 得到每个pedestrain 的平均值。

def train_cl(args, best_ade, model, train_datasets, val_datasets, replay_model="none", iters=2, batch_size=32,
             generator=None, fake_generator=None, gen_iters=0, gen_loss_cbs= list(), fake_gen_loss_cbs=list(), loss_cbs=list(), val_loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             metric_cbs=list()):
    '''  
    Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]           <nn.Module> main model to optimize across all tasks
    [train_datasets]  <list> with for each task the training <DataSet>
    [replay_mode]     <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]        <str>, choice from "task", "domain" and "class"
    [iters]           <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]       None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]           <list> of call-back functions to evaluate training-progress
    '''
    # 这儿的 generator 才是要SGR， model 是 main model
    # Set model in training-mode
    model.train()
    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None
    optimizer=None
    memory_batch = []
    U_info=None  # for first task. No replay,only main forward.
    obs_traj_record = None 
    tasks_ = None # record which task replayed data belongs to 

    # global x_rel_val, y_rel_val, seq_start_end_val

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())    # 将初始model参数存入buffer用于调用。

    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):  # 1 means index start from 1 :  so task is 1 2 3 4
        # if task==4:
        #     continue
        training_dataset = data_loader(args, train_dataset, args.batch_size)
        batch_num = len(training_dataset)       # 是个iterable，总共有多少个batch

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None   # -> for Domain-IL scenario, always all classes are active

        # Reset state of optimizer(s) for every task (if requested) todo

        # Initialize # iters left on current data-loader(s)
        # iters_left = iters_left_previous = 1
        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters_to_use + 1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters + 1))
        # replay previous data for validation
        if fake_generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters * batch_num + 1))
        if args.val:
            x_rel_val = None
            y_rel_val = None
            seq_start_end_val = None

        ##-----REPLAYED BATCH------##
        if not Exact and not Generative and not Current:
            x_rel_ = y_rel_ = seq_start_end_ = x_rel_gr = y_rel_gr = seq_start_end_gr = None  # -> if no replay

        # record losses after each epoch! For combined_model RTF
        losses_dict_main_epoch = {'loss_total':[], 'loss_current':[], 'loss_replay':[], 'pred_traj':[], 'pred_traj_r':[],'reconL':[],'variatL':[],'predL':[],'reconL_r':[],'variatL_r':[],'predL_r':[]}
        # run epoch
        for epoch in range(1, iters_to_use+1):    # 是对每个task的dataset 独立的。 每个task 的epoch
            losses_dict_main = {'loss_total':[], 'loss_current':[], 'loss_replay':[], 'pred_traj':[], 'pred_traj_r':[],'reconL':[],'variatL':[],'predL':[],'reconL_r':[],'variatL_r':[],'predL_r':[]}
            losses_dict_generative = {'loss_total':[], 'reconL':[], 'variatL':[], 'reconL_r':[], 'variatL_r':[]}
            for batch_index, batch in enumerate(training_dataset):  # 这个dataset 其实是dataloader的类，所以是iterable。
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    pred_traj_gt,
                    obs_traj_rel,
                    pred_traj_gt_rel,
                    non_linear_ped,
                    loss_mask,
                    seq_start_end,
                ) = batch

                #-------------Collect data----------------#
                ##------CURRENT BATCH-------##
                # y = y - class_per_task*(task-1) if scenario="task" else y    # --> ITL: adjust y-targets
                # print('\nout', len(out))
                x_rel = obs_traj_rel
                y_rel = pred_traj_gt_rel
                seq_start_end =seq_start_end
                loss_mask = loss_mask
                # x_rel, y_rel = out[2].to(device), out[3].to(device)                            # --> transfer them to correct device
                # seq_start_end = out[6].to(device)
                # loss_mask = out[5].to(device)

                # ------ Exact Replay ----- #   我觉着是 experience replay， 重放
                if Exact:   # default = False, will skip this replay, but after first task finish, set Exact = True, then we can do replay
                    if task == 2:   # task start from index 1,.. 
                        from helper.memory_eth import memory_buff
                        # memory_seq_dset = memory_buff(args)
                        # memory_seq_loader = iter(data_loader(args, memory_seq_dset, args.replay_batch_size, shuffle=True))
                        loader_eth, batch_eth = memory_buff(args)
                        memory_seq_loader = iter(loader_eth)
                        memory_seq = next(memory_seq_loader)
                        memory_batch.append(batch_eth)

                    if task == 3:
                        from helper.memory_eth_ucy import memory_buff
                        loader_ucy, batch_ucy = memory_buff(args, memory_batch[0])
                        memory_seq_loader = iter(loader_ucy)
                        memory_seq = next(memory_seq_loader)
                        memory_batch.append(batch_ucy)

                    if task == 4:
                        from helper.memory_eth_ucy_ind import memory_buff
                        loader_ind, batch_ind = memory_buff(args, memory_batch[0], memory_batch[1])
                        memory_seq_loader = iter(loader_ind)
                        memory_seq = next(memory_seq_loader)
                        memory_batch.append(batch_ind)
                        # iters_to_use = 100

                    # replay traj for GR
                    x_rel_ = memory_seq[2].cuda()
                    y_rel_ = memory_seq[3].cuda()
                    seq_start_end_ = memory_seq[4].cuda()

                # ----Generative / Current Replay----#
                if Generative:
                    # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                    # x_ = x if Current else previous_generator.sample(out.to(device))  # 64*500  ## when feedback is True ,here the previous_generator is actually previous model! #YZ
                    replay_data_loader = iter(
                        data_loader(args, train_dataset, args.replay_batch_size))  # replay_batch_size=64 和 current dataset的一样   # 用的是当前数据集！
                    replay_out = next(replay_data_loader)   # 扔进来一个batch
                    obs_traj_record=replay_out[0].to(device)              
                    if args.hidden ==False:  # no hidden replay. sample as normal.
                        if args.replay_model == 'lstm':
                            # replay_traj = previous_generator.sample(x_rel, obs_traj, seq_start_end)  # previous generator就是 deepcopy的前一个generator
                            replay_traj = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device),
                                                                    replay_out[6].to(device),task_id_cur=task )  # 2,3,4
                            x_rel_ = replay_traj[1]
                            x_ = replay_traj[0]  #absolute position is not used . 
                            seq_start_end_ = replay_traj[2]
                            # gate based method
                            if args.gate_decoder == True:    
                                tasks_=(replay_traj[3])
                            else:
                                tasks_=None

                            # new：add y_rel_   otherwise, train a batch will raise error!
                            #y_rel_ = previous_model(x_rel_, seq_start_end_)
                        if args.replay_model == 'vrnn':
                            # replay_traj = previous_generator.sample(args.obs_len, args.replay_batch_size)
                            replay_traj = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device),
                                                                    replay_out[6].to(device))
                            x_rel_ = replay_traj.cuda()
                            seq_start_end_ = seq_start_end
                            # previous_model.eval()
                            #y_rel_ = previous_model(x_rel_, seq_start_end_)
                    
                    else: # Internal replay, sample result is hidden state.  ##YZ
                        if args.replay_model == 'lstm':
                            decode_h,U_info = previous_generator.sample(replay_out[2].to(device), replay_out[0].to(device),
                                                                    replay_out[6].to(device),task_id_cur=task)  #here previous_generator is actaully combined model's full part.
                            x_rel_ = decode_h
                            _,seq_start_end_,tasks_ = U_info
                        

                        if args.replay_model == 'vrnn':
                            pass # didn't use VRNN to test internal replay.

                    if args.replay_model == "condition":    # 之前的SOTA 方法 condition generative replay
                        # memory_seq = [obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end]
                        # replay_traj = [traj, traj_rel, seq_start_end]
                        if task == 2:
                            from helper.memory_eth import memory_buff
                            # memory_seq_dset = memory_buff(args)
                            # memory_seq_loader = iter(data_loader(args, memory_seq_dset, args.replay_batch_size, shuffle=True))
                            loader_eth, batch_eth = memory_buff(args)
                            memory_seq_loader = iter(loader_eth)
                            memory_seq = next(memory_seq_loader)
                            memory_batch.append(batch_eth)

                        if task == 3:
                            from helper.memory_eth_ucy import memory_buff
                            loader_ucy, batch_ucy = memory_buff(args, memory_batch[0])
                            memory_seq_loader = iter(loader_ucy)
                            memory_seq = next(memory_seq_loader)
                            memory_batch.append(batch_ucy)

                        if task == 4:
                            from helper.memory_eth_ucy_ind import memory_buff
                            loader_ind, batch_ind = memory_buff(args, memory_batch[0], memory_batch[1])
                            memory_seq_loader = iter(loader_ind)
                            memory_seq = next(memory_seq_loader)
                            memory_batch.append(batch_ind)
                            # iters_to_use = 100

                        # replay traj for GR
                        x_rel_ = memory_seq[2].cuda()
                        y_rel_ = memory_seq[3].cuda()
                        seq_start_end_ = memory_seq[4].cuda()
                        
                    previous_model.eval()
                    if args.feedback==True:
                        y_rel_ =previous_model.infer(x_rel_, seq_start_end_ )  ##YZ original main have no infer function
                    else:
                        y_rel_ =previous_model(x_rel_, seq_start_end_ )
                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if Generative and x_rel_ is not None:
                    x_rel_.detach_()
                    x_rel_ = x_rel_.detach()
                    x_rel_ = Variable(x_rel_.data, requires_grad=True)
                    y_rel_.detach_()
                    y_rel_ = y_rel_.detach()
                    y_rel_ = Variable(y_rel_.data, requires_grad=True)

                # ----> Train Main model
                if batch_index <= iters*batch_num:    # index是指示在这个dataset中的第多少个batch，从这里限制每个task每一个epoch训练次数不超过iters

                    # Train the main model with this batch     ## ?? train a model 会用 surrogate_loss（忽略） , first task 怎么解决有的参数不存在的问题？？
                    loss_dict_main = model.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_,U_info=U_info, loss_mask=loss_mask, rnt=1./task,obs_traj_record=obs_traj_record,task=task,tasks_=tasks_)  # obs_seq_list=>obs_traj
                    main_loss_file = open("{}/loss_main_model_{}_{}_{}_{}.txt".format(args.r_dir, args.iters, args.batch_size, args.replay, args.val_class), 'a')   # 字符 'a' 表示以附加（append）模式打开文件
                    main_loss_file.write('{}: {}\n'.format(batch_index, loss_dict_main['loss_total']))  ## ?? 这里是在记录loss日志
                    main_loss_file.close()
                    losses_dict_main['loss_total'].append(loss_dict_main['loss_total'])
                    losses_dict_main['loss_current'].append(loss_dict_main['loss_current'])
                    losses_dict_main['loss_replay'].append(loss_dict_main['loss_replay'])
                    losses_dict_main['pred_traj'].append(loss_dict_main['pred_traj'])
                    losses_dict_main['pred_traj_r'].append(loss_dict_main['pred_traj_r'])
                    # to log loss
                    if args.feedback==True:
                        losses_dict_main['reconL'].append(loss_dict_main['reconL'])
                        losses_dict_main['variatL'].append(loss_dict_main['variatL'])
                        losses_dict_main['predL'].append(loss_dict_main['predL'])
                        losses_dict_main['reconL_r'].append(loss_dict_main['reconL_r'])
                        losses_dict_main['variatL_r'].append(loss_dict_main['variatL_r'])
                        losses_dict_main['predL_r'].append(loss_dict_main['predL_r'])

                    # Update running parameter importance estimates in W
                    if isinstance(model, ContinualLearner) and (model.si_c > 0):
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    W[n].add_(-p.grad * (p.detach() - p_old[n]))  ##跳过不关心?? 如何解决self.surrogate_loss() 调用首个task问题，以及W 的问题。
                                p_old[n] = p.detach().clone()

                # # -----> Train Fake Generator
                # if fake_generator is not None:
                #     # Train the generator with this batch
                #     loss_dict_fake_generative = fake_generator.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, rnt=1./task)



                # -----> Train Generator
                if generator is not None and batch_index <= gen_iters*batch_num:   # 这是说明，同一次batch经过时，先train model 再train generator，有可能出现model停了，generator继续train的情况（dataloader没放完的情况）
                    # when replay through feedback, generator is None.##YZ
                    # Train the generator with this batch
                    loss_dict_generative = generator.train_a_batch(x_rel, y_rel, seq_start_end, x_=x_rel_, y_=y_rel_, seq_start_end_=seq_start_end_, rnt=1./task,task=task,tasks_=tasks_)
                    losses_dict_generative['loss_total'].append(loss_dict_generative['loss_total'])
                    losses_dict_generative['reconL'].append(loss_dict_generative['reconL'])
                    losses_dict_generative['variatL'].append(loss_dict_generative['variatL'])
                    losses_dict_generative['reconL_r'].append(loss_dict_generative['reconL_r'])
                    losses_dict_generative['variatL_r'].append(loss_dict_generative['variatL_r'])
            
            # record each epoch losses. each element means loss of all batch losses of one epoch.
            if args.feedback == True:
                import numpy as np
                losses_dict_main_epoch['loss_total'].append(np.mean(losses_dict_main['loss_total'])) 
                losses_dict_main_epoch['loss_current'].append(np.mean(losses_dict_main['loss_current']))
                losses_dict_main_epoch['loss_replay'].append(np.mean(losses_dict_main['loss_replay']))
                losses_dict_main_epoch['reconL'].append(np.mean(losses_dict_main['reconL']))
                losses_dict_main_epoch['variatL'].append(np.mean(losses_dict_main['variatL']))
                losses_dict_main_epoch['predL'].append(np.mean(losses_dict_main['predL']))
                losses_dict_main_epoch['reconL_r'].append(np.mean(losses_dict_main['reconL_r']))
                losses_dict_main_epoch['variatL_r'].append(np.mean(losses_dict_main['variatL_r']))
                losses_dict_main_epoch['predL_r'].append(np.mean(losses_dict_main['predL_r']))
                

            if args.val:   # 一个epoch后。前面是对一个dataset的 多个iteration 循环训练，练好了后，validate / 现在还都是在同一个task下面
                if args.val_class == 'current':    #args.val_class :  whether use current or previous task validation data
                    val_dataset = data_loader(args, val_datasets[task-1], args.batch_size)  # task 是index(从1开始)，要指dataset list中的哪个dataset，所以得减一对应取切片。
                    ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                    # save val loss
                    val_loss_file = open("{}/loss_val_{}_{}_{}_{}.txt".format(args.r_dir, args.iters, args.batch_size, args.replay, args.val_class), 'a')
                    val_loss_file.write('{}: {}\n'.format(epoch, loss_val))
                    val_loss_file.close()
                    loss_val_dict_main = {'loss_val': loss_val}
                    for val_loss_cb in val_loss_cbs:
                      if val_loss_cb is not None:
                         val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                    ade_val = ade_current
                    is_best = ade_val < best_ade
                    best_ade = min(ade_val, best_ade)
                    if is_best:
                        previous_model = copy.deepcopy(model)
                        # file_dir = os.path.dirname(__file__) + "/chekpoint"  # I am not linux
                        file_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
                        if os.path.exists(file_dir) is False:
                           os.mkdir(file_dir)
                        filename = os.path.join(file_dir,
                                               "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                   method=args.method, replay=args.replay, task=task,
                                                   order=args.dataset_order, batch_size=args.batch_size,
                                                   seed=args.seed, epoch=epoch,
                                                   val=args.val, val_class=args.val_class,
                                                   si=args.si, si_c= args.si_c))
                        torch.save(model.state_dict(), filename)
                        shutil.copyfile(filename, "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                   method=args.method, replay=args.replay, task=task,
                                                   order=args.dataset_order, batch_size=args.batch_size,
                                                   seed=args.seed,
                                                   val=args.val, val_class=args.val_class,
                                                   si=args.si, si_c= args.si_c))
                if args.val_class == 'all':
                    if generator is None:
                        val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            #file_dir = os.path.dirname(__file__) + "/chekpoint"
                            file_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            checkpoint={'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': model.optimizer.state_dict()}
                            torch.save(checkpoint, filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                            if args.feedback == True:
                                previous_model = copy.deepcopy(model)
                            else: # when use origianl SGR, problem comes out, bug say model have generator ,cannot pickle,cannot deepcopy
                                from main_model.encoder import Predictor
                                previous_model = Predictor(
                                                        obs_len=args.obs_len,
                                                        pred_len=args.pred_len,
                                                        traj_lstm_input_size=args.traj_lstm_input_size,
                                                        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                                                        traj_lstm_output_size=args.traj_lstm_output_size
                                                    ).to(device)
                                previous_model.optim_type = args.optimizer
                                previous_model.optim_list = [
                                {'params': filter(lambda p: p.requires_grad, previous_model.parameters()), 'lr': args.lr},
                                ]
                                
                                if previous_model.optim_type in ("adam", "adam_reset"):
                                    previous_model.optimizer = optim.Adam(previous_model.optim_list, lr=args.lr)
                                elif previous_model.optim_type == "sgd":
                                    previous_model.optimizer = optim.SGD(previous_model.optim_list)
                                check_point = torch.load(filename)
                                previous_model.load_state_dict(check_point['model_state_dict'])
                                previous_model.optimizer.load_state_dict(check_point['optimizer_state_dict'])
                            


                    else:
                        if task >= 2:
                            ade_previous = 0
                            val_dataset = data_loader(args, val_datasets[task -1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            for i in range(task - 1):
                                val_dataset_ = data_loader(args, val_datasets[i], args.batch_size)
                                ade_, _ = utils.validate_cl(args, model, val_dataset_, epoch)  
                                ade_previous += ade_      
                        else:
                            val_dataset = data_loader(args, val_datasets[task -1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade     # best 每个task结束 reset 一次200. 所以依然会在进行新task时得到最好的model
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            #file_dir = os.path.dirname(__file__) + "/chekpoint"
                            file_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            checkpoint={'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': model.optimizer.state_dict()}
                            torch.save(checkpoint, filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                            if args.feedback == True:
                                previous_model = copy.deepcopy(model)
                            else:
                                 # when use origianl SGR, problem comes out, bug say model have generator ,cannot pickle,cannot deepcopy
                                from main_model.encoder import Predictor
                                previous_model = Predictor(
                                                        obs_len=args.obs_len,
                                                        pred_len=args.pred_len,
                                                        traj_lstm_input_size=args.traj_lstm_input_size,
                                                        traj_lstm_hidden_size=args.traj_lstm_hidden_size,
                                                        traj_lstm_output_size=args.traj_lstm_output_size
                                                    ).to(device)
                                previous_model.optim_type = args.optimizer
                                previous_model.optim_list = [
                                {'params': filter(lambda p: p.requires_grad, previous_model.parameters()), 'lr': args.lr},
                                ]
                                
                                if previous_model.optim_type in ("adam", "adam_reset"):
                                    previous_model.optimizer = optim.Adam(previous_model.optim_list, lr=args.lr)
                                elif previous_model.optim_type == "sgd":
                                    previous_model.optimizer = optim.SGD(previous_model.optim_list)
                                check_point = torch.load(filename)
                                previous_model.load_state_dict(check_point['model_state_dict'])
                                previous_model.optimizer.load_state_dict(check_point['optimizer_state_dict'])

                if args.val_class == 'replay':
                    if generator is None:
                        val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                        ade_current, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                        loss_val_dict_main = {'loss_val': loss_val}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            #file_dir = os.path.dirname(__file__) + "/chekpoint"
                            file_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
                    else:
                        if task >=2:
                            ade_previous = 0
                            val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = utils.validate_cl_replay(args, model, x_rel_val, y_rel_val, seq_start_end_val)    # 这一步有bug。 这个函数衡量的是model 对replay的数据的 预测水平。
                        else:
                            val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                            ade_current, loss_val_current = utils.validate_cl(args, model, val_dataset, epoch)
                            ade_previous = 0
                        loss_val_dict_main = {'loss_val': loss_val_current}
                        for val_loss_cb in val_loss_cbs:
                            if val_loss_cb is not None:
                                val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
                        ade_val = ade_current + ade_previous
                        is_best = ade_val < best_ade
                        best_ade = min(ade_val, best_ade)
                        if is_best:
                            previous_model = copy.deepcopy(model)
                            #file_dir = os.path.dirname(__file__) + "/chekpoint"    # I am not linux system.
                            file_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
                            if os.path.exists(file_dir) is False:
                                os.mkdir(file_dir)
                            filename = os.path.join(file_dir,
                                                    "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{epoch}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                        method=args.method, replay=args.replay, task=task,
                                                        order=args.dataset_order, batch_size=args.batch_size,
                                                        seed=args.seed, epoch=epoch,
                                                        val=args.val, val_class=args.val_class,
                                                        si=args.si, si_c=args.si_c))
                            torch.save(model.state_dict(), filename)
                            shutil.copyfile(filename,
                                            "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                method=args.method, replay=args.replay, task=task,
                                                order=args.dataset_order, batch_size=args.batch_size,
                                                seed=args.seed,
                                                val=args.val, val_class=args.val_class,
                                                si=args.si, si_c=args.si_c))
            else:
                val_dataset = data_loader(args, val_datasets[task - 1], args.batch_size)
                _, loss_val = utils.validate_cl(args, model, val_dataset, epoch)
                loss_val_dict_main = {'loss_val': loss_val}
                for val_loss_cb in val_loss_cbs:
                    if val_loss_cb is not None:
                        val_loss_cb(progress, epoch, loss_val_dict_main, task=task)
            
            

            # Main model
            # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
            # if args.val:
            #     model = copy.deepcopy(previous_model)

            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, epoch, losses_dict_main, task=task)
            if args.val:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(previous_model, epoch, task=task)
            else:
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, epoch, task=task)
            if model.label == "VAE":
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(model, epoch, task=task)

            # # Generative model
            # # Fire callbacks on each iteration
            # for loss_cb in fake_gen_loss_cbs:
            #     if loss_cb is not None:
            #         loss_cb(progress_gen, epoch, loss_dict_fake_generative, task=task)
            # for sample_cb in sample_cbs:
            #     if sample_cb is not None:
            #         sample_cb(fake_generator, epoch, task=task)

            # Generative model
            # Fire callbacks on each iteration
            for loss_cb in gen_loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress_gen, epoch, losses_dict_generative, task=task)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(generator, epoch, task=task)

        # ----> UPON FINISHING EACH TASK...
        # draw after a task finish all epoch
        if args.feedback == True:
            from BI_utils import visualize
            visualize.visualize_loss_epoches(losses_dict_main_epoch,task) # every task, 2 images, each have 4 subplots.

        # Close progress-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        if args.val:
            model = copy.deepcopy(previous_model)    # 所以现在练完 综合最好的model 就会深拷贝成为model，即出现新的model（与之前的浅拷贝来自main的model 不再关联）
        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(model, iters, task=task)

        if args.val is False:
            # REPLAY: update source for replay
            previous_model = copy.deepcopy(model)
            file_dir = os.path.dirname(__file__)
            filename = os.path.join(file_dir, "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(method=args.method, replay=args.replay, task=task, order=args.dataset_order, batch_size=args.batch_size, seed=args.seed, val=args.val, val_class=args.val_class, si=args.si, si_c=args.si_c))
            torch.save(model.state_dict(), filename)

        best_ade = 200
        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c > 0):
            model.update_omega(W, model.epsilon)
        if replay_model == 'generative':
            Generative = True
            #previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
            previous_generator = previous_model if args.feedback else copy.deepcopy(generator).eval()  # RTF mode, we use combined model to sample and get output. #YZ
        # EXEMPLARS: update exemplar sets
        if replay_model == "exemplars":
            # based on this dataset, construct new exemplar-set for this class
            # model.construct_exemplar_set(dataset=train_dataset, n=model.memory_budget)
            # model.compute_means = True
            # previous_dataset = model.exemplar_sets
            # previous_datasets.append(previous_dataset)
            Exact=True

        # Logging loss in different task/ after epochs
        if args.feedback == True:
            file_dir = os.path.dirname(__file__)
            output_file = os.path.join(file_dir,'loss_values_{val_class}_{method}.txt'.format(val_class=args.val_class,method = args.method ))
            if task == 1:
                with open(output_file,'w') as file:  # Should I add average? or last several times?
                    file.write(f"Task:{task}\n")
                    file.write('Current_loss:{}\n'.format(losses_dict_main['loss_current']))
                    file.write('Current_loss_predict:{}\n'.format(losses_dict_main['predL']))
                    file.write('Current_loss_variant:{}\n'.format(losses_dict_main['variatL']))
                    file.write('Current_loss_reconstruct:{}\n'.format(losses_dict_main['reconL']))
                    file.write('Replay_loss:{}\n'.format(losses_dict_main['loss_replay']))
                    file.write('Replay_loss_predict:{}\n'.format(losses_dict_main['predL_r']))
                    file.write('Replay_loss_variant:{}\n'.format(losses_dict_main['variatL_r']))
                    file.write('Replay_loss_reconstruct:{}\n'.format(losses_dict_main['reconL_r']))
                    file.write("\n") 
            else:
                with open(output_file,'a') as file:
                    file.write(f"Task:{task}\n")
                    file.write('Current_loss:{}\n'.format(losses_dict_main['loss_current']))
                    file.write('Current_loss_predict:{}\n'.format(losses_dict_main['predL']))
                    file.write('Current_loss_variant:{}\n'.format(losses_dict_main['variatL']))
                    file.write('Current_loss_reconstruct:{}\n'.format(losses_dict_main['reconL']))
                    file.write('Replay_loss:{}\n'.format(losses_dict_main['loss_replay']))
                    file.write('Replay_loss_predict:{}\n'.format(losses_dict_main['predL_r']))
                    file.write('Replay_loss_variant:{}\n'.format(losses_dict_main['variatL_r']))
                    file.write('Replay_loss_reconstruct:{}\n'.format(losses_dict_main['reconL_r']))
                    file.write("\n") 
            




