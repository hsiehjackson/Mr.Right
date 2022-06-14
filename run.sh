CUDA_VISIBLE_DEVICES=7 python main.py --num_gpus 1 --num_workers 32 --wandb_task_name "Wiki_Test_Upload" --test_output output_wiki_albef.json --batch_size 16 --pretrain METER --embeds_feats avg --save_checkpoint /data/chengping/checkpoints --neg_matching --pl_checkpoint ''

