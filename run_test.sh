# python main.py --mode test --pl_checkpoint  checkpoints/Wiki_final\ -\ ALBEF\ +\ CL\ -\ best\ R1/version_None/checkpoints/loss=0.80-val_loss=3.57-r1=44.00.ckpt --test_output output_wiki_albef.json --batch_size 32

# python main.py --mode test --pl_checkpoint checkpoints/Wiki_final\ -\ ViLT\ +\ CL\ +\ NM\ +\ CP\ -\ best\ R1/version_None/checkpoints/loss=5.84-val_loss=3.32-r1=73.80.ckpt --test_output output_wiki_vilt.json --batch_size 32 --pretrain ViLT

# python main.py --mode test --pl_checkpoint checkpoints/Wiki_final\ -\ ViLT\ +\ CL\ +\ NM\ +\ CP\ -\ random\ doc/version_None/checkpoints/loss=8.20-val_loss=3.29-r1=69.70.ckpt --test_output output_wiki_vilt.json --batch_size 32 --pretrain ViLT

# CUDA_VISIBLE_DEVICES=5,6 python main.py --num_gpus 2  --mode test --wandb_task_name "test" --test_output output_wiki_meter.json --batch_size 32 --pretrain METER --pl_checkpoint /data/chengping/checkpoints/multimodalembedding/Wiki_400k\ -\ METER\ \(avg\)\ +\ CL\ +\ NMv1\ +\ \(-CP\)\ +\ RDT\ +\ lr\ 5e-6\ +\ b\ 30/checkpoints/loss\=2.77-val_loss\=0.64-r1\=79.06.ckpt 

# CUDA_VISIBLE_DEVICES=0 python main.py --num_gpus 1  --mode test --wandb_task_name "test" --test_output output_wiki_meter.json --batch_size 128 --pretrain METER --pl_checkpoint /data/chengping/checkpoints/multimodalembedding/2fnnqd32/checkpoints/loss\=2.66-val_loss\=4.95-multi_r1\=32.00.ckpt

CUDA_VISIBLE_DEVICES=1 python main.py --num_gpus 1  --mode test --wandb_task_name "test" --test_output output_wiki_vilt.json --batch_size 128 --pretrain ViLT --pl_checkpoint /data/chengping/checkpoints/multimodalembedding/ecsz1tny/checkpoints/loss=2.11-val_loss=3.35-multi_r1=53.00.ckpt

# CUDA_VISIBLE_DEVICES=1 python main.py --num_gpus 1  --mode test --wandb_task_name "test" --test_output output_wiki_albef.json --batch_size 128 --pretrain ALBEF --pl_checkpoint /data/chengping/checkpoints/multimodalembedding/35pw20pv/checkpoints/loss=2.70-val_loss=4.14-multi_r1=42.00.ckpt
# ,3,4,5,6

# /home/u9315001/MultimodalEmbedding/checkpoints/multimodalembedding/Wiki_400k\ -\ METER\ \(avg\)\ +\ CL\ +\ NMv1\ +\ \(-CP\)\ +\ RDT\ +\ lr\ 5e-6\ +\ b\ 30/checkpoints/loss\=2.77-val_loss\=0.64-r1\=79.06.ckpt