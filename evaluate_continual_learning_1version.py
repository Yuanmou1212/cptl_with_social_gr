import argparse
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle

import torch

from data.loader import data_loader, data_dset
from main_model.encoder import Predictor
#from helper.evaluate import evaluate
from helper import evaluate
from helper import utils
from helper.utils import (
    displacement_error,
    final_displacement_error,
    l2_loss,
    int_tuple,
    relative_to_abs,
    get_dset_path,
)

### 需要明确的是于batch learning的区别，首先是best model 文件保存在当前文件夹下，名字要和
### train_cl中 if args.val_class == 'replay': 对应的保存部分吻合。其次要check parser参数。
##  要改 dataset的path， 连续的task，而不是单独一次。
## 

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="ETH", help="Directory containing logging file")
parser.add_argument("--dataset_name_train", default="ETH", type=str)
parser.add_argument("--dataset_name_test", default="ETH", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=8, type=int)
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")
parser.add_argument("--batch_size", default=64, type=int)  # 200->64
parser.add_argument("--val_epoch", default=2, type=int)
augmentation_choices = ["none", "rotation"]
parser.add_argument("--aug", type=str, default='none', choices=augmentation_choices, help="whether to rotation the data")

parser.add_argument("--noise_dim", default=(8,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian")
parser.add_argument("--noise_mix_type", default="global")

# lstm
parser.add_argument("--traj_lstm_input_size", type=int, default=2, help="traj_lstm_input_size")
parser.add_argument("--traj_lstm_hidden_size", default=32, type=int)
parser.add_argument('--traj_lstm_output_size', default=32, type=int)
# gat
parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
parser.add_argument("--hidden-units", type=str, default="16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument("--graph_network_out_dims", type=int, default=32, help="dims of every node after through GAT module")
parser.add_argument("--graph_lstm_hidden_size", default=32, type=int)
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
model_choices = ["lstm", "gat"]
parser.add_argument('--main_model', default='lstm', type=str, choices=model_choices, help="the main model of CL and BL")

parser.add_argument("--num_samples", default=20, type=int)
parser.add_argument("--dset_type", default="test", type=str)
parser.add_argument("--resume", default="model_best.pth.tar", type=str, metavar="PATH", help="path to latest checkpoint (default: none)",)

# corresponding parse
method_choices = ['batch_learning', 'continual_learning']
parser.add_argument('--method', type=str, default='continual_learning', choices=method_choices)
replay_choices = ['offline', 'exact', 'generative', 'none', 'current', 'exemplars']  # ER is exact or none. check
parser.add_argument('--replay', type=str, default='generative', choices=replay_choices)
dataset_choices = ['pedestrian', 'vehicle', 'interaction']
parser.add_argument('--dataset_order', type=str, default='pedestrian', choices=dataset_choices)
parser.add_argument('--val', action='store_true', help="use validation data")  # 需要输入 --val 
class_choices = ['current', 'all', 'replay']
parser.add_argument('--val_class', default='current', type=str, choices=class_choices, help='whether use current or previous task validation data')
parser.add_argument('--si', action='store_true', help="use 'Synaptic Intelligence' (Zenke, Poole et al, 2017)")
parser.add_argument('--c', type=float, dest="si_c", help="--> SI: regularisation strength")


def get_model(checkpoint):
    if args.main_model == "lstm":
        from main_model.encoder import Predictor
        model = Predictor(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size,
            traj_lstm_output_size=args.traj_lstm_output_size
        )
    if args.main_model == "gat":
        n_units = (
                [args.traj_lstm_hidden_size]
                + [int(x) for x in args.hidden_units.strip().split(",")]
                + [args.graph_lstm_hidden_size]
        )
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        from main_model.encoder_gat import Predictor
        model = Predictor(
            obs_len=args.obs_len, pred_len=args.pred_len, traj_lstm_input_size=args.traj_lstm_input_size,
            traj_lstm_hidden_size=args.traj_lstm_hidden_size, traj_lstm_output_size=args.traj_lstm_output_size,
            n_units=n_units, n_heads=n_heads, graph_network_out_dims=args.graph_network_out_dims,
            dropout=args.dropout, alpha=args.alpha, graph_lstm_hidden_size=args.graph_lstm_hidden_size
        )

    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    return model



def main(args):   
    #checkpoint_path = os.path.join(os.path.abspath(args.log_dir), "{}_{}_{}_best.pth.tar".format(args.main_model, args.dataset_name_train, args.aug))
    #checkpoint = torch.load(checkpoint_path)
    
    for i in range(1,5):
        ## preprare test dataset.
        task = i # Here we can add for loop to go through all task finished model.
        
        if args.dataset_order == 'pedestrian':    # 为了简化计算
            train_order = ['ETH', 'UCY', 'inD', 'INTERACTION']
            val_order = ['ETH', 'UCY', 'inD', 'INTERACTION']
            test_order = ['ETH', 'UCY', 'inD', 'INTERACTION']
                
        if args.dataset_order == 'vehicle':
            train_order = ['highD', 'inD', 'rounD']
            val_order = ['highD', 'inD', 'rounD']
            test_order = ['highD', 'inD', 'rounD']
        if args.dataset_order == 'interaction':
            train_order = ['MT', 'SR', 'LN', 'OF']
            val_order = ['MT', 'SR', 'LN', 'OF']
            test_order = ['MT', 'SR', 'LN', 'OF']
        
        num_index = task
        train_order = train_order[:num_index]
        val_order = val_order[:num_index]
        test_order = test_order[:num_index]

        test_datasets=[]
        print("\nInitializing test dataset")
        for i, dataset_name in enumerate(test_order):
            # load test dataset path
            test_path = utils.get_dset_path(dataset_name, "test")
            # load test dataset
            test_dset = data_dset(args, test_path)
            test_loader = data_loader(args, test_dset, args.batch_size)
            print("dataset: {} | test trajectories: {}".format(dataset_name, (test_dset.obs_traj.shape[0])))
            test_datasets.append(test_loader)


        ## Load parameters from saved files.
        checkpoint_path = os.path.join(os.path.dirname(__file__), "{method}_{replay}_{task}_model_{order}_{batch_size}_{seed}_{val}_{val_class}_{si}_{si_c}.path".format(
                                                    method=args.method, replay=args.replay, task=task,
                                                    order=args.dataset_order, batch_size=args.batch_size,
                                                    seed=args.seed,
                                                    val=args.val, val_class=args.val_class,
                                                    si=args.si, si_c=args.si_c))
        checkpoint = torch.load(checkpoint_path)
        
        
        model = get_model(checkpoint)   
        path = get_dset_path(args.dataset_name_test, args.dset_type)
        
        ## evaluate
        ades = []
        fdes = []
        for i in range(task):
            ade,fde = evaluate.validate(model,test_datasets[i])
            ades.append(ade)
            fdes.append(fde)
        average_ades = sum(ades) / task
        average_fdes = sum(fdes) / task


        # data_set = data_dset(args,path)
        # loader = data_loader(args, data_set,args.batch_size) #  补上缺失的 args.batch_size
        # ade, fde = evaluate(loader, model)
        # d = {'training dataset': args.dataset_name_train, 'testing dataset': args.dataset_name_test, 'Pred len': args.pred_len,
        #      'ADE': ade, 'FDE': fde}
        # utils.save_dict(d, "batch_learning_{}_{}_{}_{}".format(args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
        # utils.save_dict_txt(d, "batch_learning_{}_{}_{}_{}".format(args.dataset_name_train, args.dataset_name_test, args.aug, args.main_model))
        # print(
        #     "Train Dataset: {} | Test Dataset: {} | Pred Len: {} | ADE: {:.12f} | FDE: {:.12f}".format(
        #         args.dataset_name_train, args.dataset_name_test, args.pred_len, ade, fde
        #     )
        # )

        for i in range(task):
            print(" - Task {}: ADE {:.4f} FDE {:.4f}".format(i+1, ades[i], fdes[i]))
        print("==> Average precision over all {} tasks: ADE {:.4f} FDE {:.4f}".format(task, average_ades, average_fdes))
  
if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)


