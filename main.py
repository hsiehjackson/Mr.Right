import yaml
import os
import utils
import warnings
from argparse import ArgumentParser
from torch import nn
from pytorch_lightning import Trainer,seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from data.data_module import TextToMultiDataModule
from pltrainer import TextToMultiTrainer
from functools import partial
from models.model import TextToMultiModel
from transformers import (
    BertTokenizer, RobertaTokenizerFast
)
warnings.filterwarnings("ignore")

def main(args,config):
    seed_everything(config.seed)
    
    # tokenizer
    if args.pretrain == "ALBEF" or args.pretrain == "ViLT":
        tokenizer = BertTokenizer.from_pretrained(
            config.text_encoder,
            cache_dir= args.cache_dir,
        )
    elif args.pretrain == "METER":
        tokenizer = RobertaTokenizerFast.from_pretrained(
            config.text_encoder,
            cache_dir= args.cache_dir,
        )

    # dataset
    print("Create Dataset")
    data_module = TextToMultiDataModule(args,config,tokenizer)
    if args.mode == "test":
        data_module.prepare_data(test=config['test_file'],document=config['document'])
    else:
        data_module.prepare_data(train=config['train_file'],val=config['val_file'],test=config['test_file'],document=config['document'])
    data_module.setup()  

    # mutli model
    print("Create multi modal")
    model = TextToMultiModel(tokenizer=tokenizer,config=config,args=args) 

    pltrainer = TextToMultiTrainer(args,config,model,tokenizer)

    # logger
    wandb_logger = WandbLogger(name=args.wandb_task_name,project="multimodalembedding", entity=args.wandb_entity)
    checkpoint_callback = ModelCheckpoint(
        filename= '{loss:.2f}-{val_loss:.2f}-{multi_r1:.2f}', 
        save_top_k=3, 
        verbose=False,
        monitor='multi_r1', 
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    
    trainer_config = {
        "default_root_dir": args.save_checkpoint,
        "fast_dev_run": False,
        # "gradient_clip_val": config.gradient_clip_value,
        # "replace_sampler_ddp":False,
        "strategy": "ddp",
        "logger": wandb_logger,
        "gpus": args.num_gpus,
        "max_epochs": config["schedular"]["epochs"],
        # "max_steps": config["schedular"]["max_steps"],
        "auto_scale_batch_size": 'binsearch',
        "progress_bar_refresh_rate": 1,
        "precision": 16,
        "check_val_every_n_epoch": 10,
        "log_every_n_steps": 1,
        "flush_logs_every_n_steps": 1,
        "callbacks":[checkpoint_callback, lr_monitor],
    }
    trainer = Trainer(**trainer_config)
    if args.mode == "train":
        trainer.fit(pltrainer, data_module,ckpt_path=args.pl_checkpoint)
    elif args.mode == "test":
        trainer.test(pltrainer, data_module,ckpt_path=args.pl_checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wandb_task_name', default='testing')
    parser.add_argument('--wandb_entity', default='multimodalembedding')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument("--config", default="configs/ALBEF.yaml", type=str)
    parser.add_argument("--cache_dir", default="cache/", type=str)
    parser.add_argument('--log_dir', default='logs/',type=str)
    parser.add_argument("--mode", default="train", type=str,choices=['train','test'],help='choose your mode')
    parser.add_argument("--pretrain", default="ALBEF", type=str,choices=['ALBEF','ViLT','MDETR','METER'],help='choose pretrain work')
    parser.add_argument("--embeds_feats", default="avg", type=str,choices=['cls','avg','iavg_tcls'],help='how to deal with text and image embeddings')
    parser.add_argument("--pickle_output", default="./", type=str,help='directory of testing pickle files')
    parser.add_argument("--test_output", default="output.json", type=str,help='json files of testing result')
    parser.add_argument("--save_checkpoint", default="checkpoints", type=str)
    parser.add_argument('--pl_checkpoint', default=None,type=str,help='Load pytorch lightning checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,help='The batch size of each dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of workers in the DataLoader')
    parser.add_argument('--shuffle', type=bool, default=True,help='Whether shuffle dataloader')
    parser.add_argument('--ctx_prediction', action='store_true', help='Whether do context prediction')
    parser.add_argument('--neg_matching', action='store_true', help='Whether do negative matching')
    parser.add_argument('--neg_matchingv2', action='store_true', help='Whether do negative matching version2')
    parser.add_argument('--test_rank', default=10, type=int, help='Step1. Contrastive -> rank -> Step2. Matching')
    parser.add_argument('--re_ranking', action='store_true', help='Whether do re ranking for matching')
   
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)
    if not os.path.exists(args.save_checkpoint):
        os.makedirs(args.save_checkpoint, exist_ok=True)

    if args.pretrain == "ALBEF":
        args.config = "configs/ALBEF.yaml"
    elif args.pretrain == "ViLT":
        args.config = "configs/ViLT.yaml"
    elif args.pretrain == "MDETR":
        args.config = "configs/MDETR.yaml"
    elif args.pretrain == "METER":
        args.config = "configs/METER.yaml"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = utils.AttrDict(config)
    print(args)
    print(config)
    main(args,config)