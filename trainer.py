import os
import sys
import copy
import logging

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./etc')
from utils import get_model_list

logger = logging.getLogger(__name__)

from model import (Generator)


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.gen = Generator(config['model']['gen'])
        self.gen_ema = copy.deepcopy(self.gen)

        self.model_dir = config['model_dir']
        self.config = config

        lr_gen = config['lr_gen']
        gen_params = list(self.gen.parameters())
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen,
                                        weight_decay=config['weight_decay'])

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.gen = nn.DataParallel(self.gen).to(self.device)
            self.gen_ema = nn.DataParallel(self.gen_ema).to(self.device)

    def train(self, loader, writer):
        config = self.config

        def run_epoch(epoch):
            self.gen.train()

            pbar = tqdm(enumerate(zip(loader['train_road'])), total=len(loader['train_road']))

            for it, road in pbar:
                gen_loss_total, gen_loss_dict = self.compute_gen_loss(road)
                self.gen_opt.zero_grad()
                gen_loss_total.backward()
                nn.utils.clip_grad_norm(self.gen.parameters(), 1.0)
                self.gen_opt.step()
                update_average(self.gen_ema, self.gen)

                log = "Epoch[%i/%i],  " % (epoch + 1, config['max_epochs'])
                all_losses = dict()
                for loss in [gen_loss_dict]:
                    for key, value in loss.items():
                        if key.find('total') > -1:
                            all_losses[key] = value
                log += ' '.join(['%s: [%.2f]' % (key, value) for key, value in all_losses.items()])
                pbar.set_description(log)

                if (it + 1) % config['log_every'] == 0:
                    for k, v in gen_loss_dict.items():
                        writer.add_scalar(k, v, epoch * len(loader['train_road']) + it)

        for epoch in range(config['max_epoch']):
            run_epoch(epoch)

            if (epoch + 1) % config['save_every'] == 0:
                self.save_checkpoint(epoch + 1)

    def compute_gen_loss(self, road):
        config = self.config

        road_in = road['train'].to(self.device)

        f1, f2, f3, f4, f5 = self.gen(road_in)

        loss_recon = nn.BCELoss(f1, road)
        loss_regularization = -0.5 * torch.sum(1 + 2 * f2 - f3 ** 2 - torch.exp(f2) ** 2).sum(1).mean()

        l_total = (config['rec_w'] * loss_recon
                   + config['reg_w'] * loss_regularization)

        l_dict = {'loss_total': l_total,
                  'loss_recon': loss_recon,
                  'loss_regul': loss_regularization}

        return l_total, l_dict

    @torch.no_grad()
    def test(self, road):
        config = self.config
        self.gen_ema.eval()

        f1, f2, f3, f4, f5 = self.gen_ema(road, phase='test')
        loss_recon = nn.BCELoss(f1, road)
        loss_regularization = -0.5 * torch.sum(1 + 2 * f2 - f3 ** 2 - torch.exp(f2) ** 2).sum(1).mean()

        l_total = (config['rec_w'] * loss_recon
                   + config['reg_w'] * loss_regularization)

        l_dict = {'loss_total': l_total,
                  'loss_recon': loss_recon,
                  'loss_regul': loss_regularization}

        out_dict = {
            "recon_road": f4,
            "road_gt": road
        }

        return out_dict, l_dict

    def save_checkpoint(self, epoch):
        gen_path = os.path.join(self.model_dir, 'gen_%03d.pt' % epoch)

        raw_gen = self.gen.module if hasattr(self.gen, "module") else self.gen
        raw_gen_ema = self.gen_ema.module if hasattr(self.gen_ema, "module") else self.gen_ema

        logger.info("saving %s", gen_path)
        torch.save({'gen': raw_gen.state_dict(), 'gen_ema': raw_gen_ema.state_dict()}, gen_path)

        print('Saved model at epoch %d' % epoch)

    def load_checkpoint(self, model_path=None):
        if not model_path:
            model_dir = self.model_dir
            model_path = get_model_list(model_dir, "gen")

        state_dict = torch.load(model_path, map_location=self.device)
        self.gen.load_state_dict(state_dict['gen'])
        self.gen_ema.load_state_dict(state_dict['gen_ema'])

        epochs = int(model_path[-6:-3])
        print('Load from epoch %d' % epochs)

        return epochs


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


if __name__ == '__main__':
    import argparse
    from etc.utils import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    config = get_config(args.config)
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")

    trainer = Trainer(config)

    xa = torch.randn(1, 2, 3, 4)  # re implement this
    xa_foot = torch.zeros(1, 4, 20)

    xa_in = {'road': xa}

    trainer.compute_gen_loss(xa_in)
