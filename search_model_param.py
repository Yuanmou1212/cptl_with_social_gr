import os
import torch
if __name__ == '__main__':
    file_dir = os.path.join(os.path.dirname(__file__),'model_params','BEST_MODEL')
    file_name = os.path.join(file_dir, 'BEST.path')
    check_point = torch.load(file_name)

    # 打印检查点中包含的键
    print(check_point.keys())