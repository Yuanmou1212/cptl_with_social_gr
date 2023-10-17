import torch
from helper import utils
from helper import visual_visdom


def evaluate_helper(error, seq_start_end): # error  仅有一个元素 (1，batch)组成的list/
    sum_ = 0
    error = torch.stack(error, dim=1)   # 在一个新的维度（dim1）上堆叠list 中的元素（只有一个，所以仅仅是转置）  原先的error这个list中的每一个元素按列值合并堆叠   =》 （batch size，num of batch=1）
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]       # （batch size,1） 切第一维度，没问题..
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)      # 由于每次输入的list只有一个元素，所以没用上这个min
        sum_ += _error
    return sum_



####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####

def initiate_metrics_dict(n_tasks):
    metrics_dict = {}
    metrics_dict["average_ade"] = []
    metrics_dict["average_fde"] = []
    metrics_dict["x_iteration"] = []
    metrics_dict["x_task"] = []
    metrics_dict["ade per task"] = {}
    metrics_dict["fde per task"] = {}
    for i in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(i+1)] = []
        metrics_dict["fde per task"]["task {}".format(i+1)] = []
    return metrics_dict

def intial_accuracy(model, datasets, metric_dict, test_size=None, verbose=False, no_task_mask=False):
    n_tasks = len(datasets)
    ades = []
    fdes = []

    for i in range(n_tasks):
        ade, fde = validate(model, datasets[i])
        ades.append(ade)
        fdes.append(fde)

    metric_dict["initial ade per task"] = ades
    metric_dict["initial fde per task"] = fdes
    return metric_dict


def metric_statistics(model, datasets, current_task, iteration,
                      metrics_dict=None, test_size=None, verbose=False):
    n_tasks = len(datasets)
    ades_all_classes = []
    fdes_all_classes = []
    ades_all_classes_ = []
    fdes_all_classes_ = []
    # for i in range(n_tasks):
    #     ade, fde = validate(model, datasets[i]) if (i<current_task) else (0.,0.)
    #     ades_all_classes.append(ade)
    #     fdes_all_classes.append(fde)

    for i in range(n_tasks):
        ade_, fde_ = validate(model, datasets[i])
        ades_all_classes_.append(ade_)
        fdes_all_classes_.append(fde_)

    average_ades = sum([ades_all_classes_[task_id] for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes_all_classes_[task_id] for task_id in range(current_task)]) / current_task

    for task_id in range(n_tasks):
        metrics_dict["ade per task"]["task {}".format(task_id+1)].append(ades_all_classes_[task_id])
        metrics_dict["fde per task"]["task {}".format(task_id+1)].append(fdes_all_classes_[task_id])

    metrics_dict["average_ade"].append(average_ades)
    metrics_dict["average_fde"].append(average_fdes)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen

    return metrics_dict



###---------------------------------------------------------------------------------------------------###

##-----------------------------##
##----PREDICTION EVALUATION----##
##-----------------------------##

def cal_ade_fde(pred_traj_gt, pred_traj_fake):
    ade_ = utils.displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    fde_ = utils.final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    return ade_, fde_

def evaluate(loader, predictor):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) =batch

            ade, fde = [], []   # 每次只扔一个batch 进来，所以这个list只有一个batch...
            total_traj += pred_traj_gt.size(1)
            pred_len = pred_traj_gt.size(0)   # （seq,batch(ped),position)

            for _ in range(1):
                pred_traj_fake_rel = predictor(obs_traj_rel, seq_start_end)
                pred_traj_fake = utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1]) 
                ade_, fde_ = cal_ade_fde(pred_traj_gt, pred_traj_fake)     #因为rel-》abs， 所以对比的是绝对坐标的差距。
                ade.append(ade_)  # 一个batch中 多条traj 得到一个batch的ade，fde值。（batch，1） ， （batch，1）
                fde.append(fde_)

            ade_sum = evaluate_helper(ade, seq_start_end)  # 搞半天就是把所有 每条traj（batch) 的error加起来。 转置，stack对于一个元素的list都没用。
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum) # 进来的每个是一个batch中的很多条traj 的error之和，就算一个batch的error
            fde_outer.append(fde_sum)
        ade = sum(ade_outer).item() / (total_traj * pred_len)
        fde = sum(fde_outer).item() / (total_traj)
        return ade, fde


def validate(model, dataset_name, batch_size=128, test_size=1024, verbose=True):   # main中调用，输入的是 test dataset 。 所以这个函数就是evaluate功能
    '''
    Evaluate precision (ADE and FDE) of a predictor ([model]) on [dataset].
    '''
    # test_loader = data_loader(args, test_dset, args.batch_size)   test_datasets.append(test_loader)  所以这里输入的dataset_name 是指对应的test dataset的dataloader 

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Loop over batches in [dataset]   
    '''
    print("\nInitializing test dataset")
    test_path = utils.get_dset_path("test")
    data_type = ".txt"
    _, test_loader = data_loader(args, test_path, dataset_name, data_type)
    ''' 
    ade, fde = evaluate(dataset_name, model)    # 扔进去的是dataloader

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    '''
    if verbose:
        print("\nDataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            dataset_name, args.pred_len, ade, fde
        ))
    '''
    return ade, fde  # 均值， 每个点。而不是每条路径。

def precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
              test_size=None, visdom=None, verbose=False, summary_graph=True):
    n_tasks = len(datasets)
    ades = []
    fdes = []
    for i in range(n_tasks):
        if i+1 <=current_task:
            ade, fde = validate(model, datasets[i])
            ades.append(ade)
            fdes.append(fde)
        else:
            ades.append(0)
            fdes.append(0)

    average_ades = sum([ades[task_id] for task_id in range(current_task)]) / current_task
    average_fdes = sum([fdes[task_id] for task_id in range(current_task)]) / current_task

    # Send results to visdom server
    names = ['task {}'.format(i+1) for i in range(n_tasks)]
    if visdom is not None:
        visual_visdom.visualize_scalars(
            ades, names=names, title="ADE on validation set (CL_{})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="ADE precision"
        )
        visual_visdom.visualize_scalars(
            fdes, names=names, title="FDE on validation set (CL_{})".format(visdom["graph"]),
            iteration=iteration, env=visdom["env"], ylable="FDE precision"
        )
        if n_tasks > 1 and summary_graph:
            visual_visdom.visualize_scalars(
                [average_ades], names=["ADE"], title="Average ADE on validation set (CL_{})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="ADE precision"
            )
            visual_visdom.visualize_scalars(
                [average_fdes], names=["FDE"], title="Average FDE on validation set (CL_{})".format(visdom["graph"]),
                iteration=iteration, env=visdom["env"], ylable="FDE precision"
            )







