# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'power_pred'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'WaveNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    train_data_root = './raw_data/train_data.csv'

    WINDOW_SIZE = 48
    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    
    input_size = 9
    out_size = 1
    residual_size = 3
    skip_size = 3
    dilation_cycles = 1
    dilation_depth = 4

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
#         opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
