from config import opt
import torch
import torch.nn as nn
import models
from data import Data_utility
from utils.visualize import Visualizer
from tqdm import tqdm
import joblib
from torchnet import meter



def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port=opt.vis_port)
    
    model = models.WaveNet(opt.input_size,opt.out_size,opt.residual_size,
                       opt.skip_size,opt.dilation_cycles,opt.dilation_depth)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    model.to(device)
    
    data_utility = Data_utility(opt.train_data_root,opt.WINDOW_SIZE)
    scaler = data_utility.get_scaler()
    joblib.dump(scaler,'scaler.pkl')
    
    X,Y = data_utility.get_data()
    
    criterion = nn.MSELoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr,opt.weight_decay)
    
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for i,(data,label) in tqdm(enumerate(data_utility.get_batches(X,Y,opt.batch_size))):
            
            inputs = data.to(device)
            targets = label.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            preds = preds.squeeze(2)
            loss = criterion(preds,targets)
            loss.backward()
            optimizer.step()
            
            loss_meter.add(loss.item())
            if(i+1)%opt.print_freq == 0:
                vis.plot('loss',loss_meter.value()[0])
        
        save_name = 'models/checkpoints/'+opt.model+str(epoch)+'.pth'
        model.save(save_name)
        
        if loss_meter.value()[0]>previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]

        
        
        
if __name__=='__main__':
    import fire
    fire.Fire()