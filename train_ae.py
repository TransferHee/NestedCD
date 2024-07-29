from argparse import ArgumentParser

import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from model import ReconsAE
from pl_datamodule import Shape16DataModule

def parse_config(config_path):
    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    return args

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/graph_enc_bias_dec_emd.yaml')
    parser = Shape16DataModule.add_argparse_args(parser)
    parser = ReconsAE.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    config = parse_config(args.config)

    args.batch_size = config['parameter']['batch_size']
    args.num_workers = config['parameter']['num_workers']

    dm = Shape16DataModule.from_argparse_args(args)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
    )

    if config['env']['multi_gpu']:
        trainer = Trainer.from_argparse_args(args, gpus=config['env']['gpu'],
                                             max_epochs=config['parameter']['epochs'],
                                             precision=16,
                                             accelerator='ddp',
                                             checkpoint_callback=checkpoint_callback,
                                             check_val_every_n_epoch=20)
    else:
        trainer = Trainer.from_argparse_args(args, gpus=config['env']['gpu'],
                                             max_epochs=config['parameter']['epochs'],
                                             precision=16,
                                             checkpoint_callback=checkpoint_callback,
                                             check_val_every_n_epoch=100)
    current_loss = config['model']['loss']
    print('\n============',current_loss,'=============\n')
    model = ReconsAE(config)
    
    trainer.fit(model, dm)
    
if __name__ == '__main__':
    main()
