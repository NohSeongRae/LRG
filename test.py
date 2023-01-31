import torch
import numpy as np
import os
import sys
import argparse
from trainer import Trainer
sys.path.append('./etc')
sys.path.append('./preprocess')

from utils import ensure_dirs, get_configs

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_path(config):
    config['main_dir']=os.path.join('.', config['name'])
    config['model_dir']=os.path.join(config['main_dir'], "pth")
    ensure_dirs([config['main_dir'], config['model_dir']])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='./road_gen_model/info/config.yaml',
                        help='Path to the config file.')
    parser.add_argument('--road',
                        type=str,
                        default='./datasets/norm/atlanta.csv')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./output')
    args=parser.parse_args()

    #initialize path
    cfg=get_configs(args.config)
    initialize_path(cfg)

    #make output path folder
    output_dir=args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    #input road
    road_network=args.road

    road_network_name=os.path.split(road_network)[-1].split('.')[0]
    recon_name=road_network_name+'_recon'


    #preprocess, normalization

    #Trainer
    trainer=Trainer(cfg)
    epochs=trainer.load_checkpoint()

    loss_test={}
    with torch.no_grad():
        road_data=road.to(device)

        outputs, loss_test_dict=trainer.test(road_data)
        rec=outputs["recon_road"].squeeze()
        gt=outputs["road_gt"].squeeze()


    #some map generations...

    for key in loss_test_dict.keys():
        loss=loss_test_dict[key]
        if key not in loss_test:
            loss_test[key]=[]

        loss_test[key].append(loss)

    log=f'Load epoch [{epochs}, '
    loss_test_avg=dict()
    for key, loss in loss_test.items():
        loss_test_avg[key] = sum(loss) / len(loss)
    log += ' '.join([f'{key:} [{value:}]' for key, value in loss_test_avg.items()])
    print(log)

if __name__=='__main__':
    main()