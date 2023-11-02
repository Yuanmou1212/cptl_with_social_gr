import matplotlib.pyplot as plt
import os
def visualize_loss_epoches(losses_dict_main_epoch,task):
    # each elemetn is from each epoch  (get a average loss) 
    fig1, axes1 = plt.subplots(2,2,figsize=(12,8))
    fig1.suptitle('Average Current Loss of each Epoch of Task {}'.format(task))
    axes1[0][0].plot(losses_dict_main_epoch['loss_current'])
    axes1[0][1].plot(losses_dict_main_epoch['predL'])
    axes1[1][0].plot(losses_dict_main_epoch['reconL'])
    axes1[1][1].plot(losses_dict_main_epoch['variatL'])
    
    axes1[0][0].set_title('Current Loss')
    axes1[0][0].set_xlabel('Epoch')
    axes1[0][0].set_ylabel('Loss')
    axes1[0][1].set_title('Prediction Loss')
    axes1[0][1].set_xlabel('Epoch')
    axes1[0][1].set_ylabel('pred_Loss')
    axes1[1][0].set_title('Reconstruction Loss')
    axes1[1][0].set_xlabel('Epoch')
    axes1[1][0].set_ylabel('recon_Loss')
    axes1[1][1].set_title('Variant Loss')
    axes1[1][1].set_xlabel('Epoch')
    axes1[1][1].set_ylabel('Varia_Loss')

    plt.tight_layout()
    
    file_dir = os.path.join(os.path.dirname(__file__),'..' ,"Figure_loss")
    if os.path.exists(file_dir) is False:
        os.mkdir(file_dir)
    path = os.path.join(file_dir,'loss_current_Task{}'.format(task))
    plt.savefig(path)
    plt.close(fig1)
    # clear , save replay loss picture
    plt.clf()

    fig2, axes2 = plt.subplots(2,2,figsize=(12,8))
    fig2.suptitle('Average Replay Loss of each Epoch of Task {}'.format(task))
    axes2[0][0].plot(losses_dict_main_epoch['loss_replay'])
    axes2[0][1].plot(losses_dict_main_epoch['predL_r'])
    axes2[1][0].plot(losses_dict_main_epoch['reconL_r'])
    axes2[1][1].plot(losses_dict_main_epoch['variatL_r'])

    axes2[0][0].set_title('Replay Loss')
    axes2[0][0].set_xlabel('Epoch')
    axes2[0][0].set_ylabel('Loss')
    axes2[0][1].set_title('Replay Prediction Loss')
    axes2[0][1].set_xlabel('Epoch')
    axes2[0][1].set_ylabel('pred_Loss')
    axes2[1][0].set_title('Replay Reconstruction Loss')
    axes2[1][0].set_xlabel('Epoch')
    axes2[1][0].set_ylabel('recon_Loss')
    axes2[1][1].set_title('Replay Variant Loss')
    axes2[1][1].set_xlabel('Epoch')
    axes2[1][1].set_ylabel('Varia_Loss')

    path = os.path.join(file_dir,'loss_replay{}'.format(task))
    plt.savefig(path)
    plt.close(fig2)