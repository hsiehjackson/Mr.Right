# Mr. Right: Multimodal Retrieval on Representation of ImaGe witH Text
Mr. Right is a novel retrieval dataset containing multimodal documents (images and texts) and multi-related queries. It also provides a multimodal framework for evaluation and compares with previous text-to-text retrieval models and image-text retrieval models. Dataset and model checkpoints are released.

For more details, please checkout our Mr. Right paper.

![Mr. Right Dataset](https://github.com/hsiehjackson/Mr.Right/blob/main/Mr_Right_framework.png?raw=true)

## Dataest
* <a href="https://www.dropbox.com/s/jky5dvkn6nar8mc/Mr_Right.tar.gz?dl=1"> Dataset json files</a>
    * multimodal_documents.json
        * amount: 806,357
        * id: document id
        * title: document title
        * doc_text: document text
        * img_url: document image URL
    * multimodal_pretrain_pairs.json
        * amount: 351,979
        * id: document id
        * title: document title
        * doc_text: document text
        * org_doc_text: wiki original document text (no remove snippets)
        * query_text: text-related query from auto generation
        * query_img: image-related query from auto generation
        * img_url: document image URL
    * multimodal_finetune_pairs.json
        * amount: 900
        * id: document id
        * title: document title
        * doc_text: document text
        * query_text: text-related query from human annotation
        * query_img: image-related query from human annotation
        * query_multi: mixed query from human annotation
        * img_url: document image URL
    * multimodal_val_queries.json
        * amount: 100
        * id: document id
        * title: document title
        * query_text: text-related query from human annotation
        * query_img: image-related query from human annotation
        * query_multi: mixed query from human annotation
    * multimodal_test_queries.json
        * amount: 2,047
        * id: document id
        * title: document title
        * doc_text: document text
        * query_text: text-related query from human annotation
        * query_img: image-related query from human annotation
        * query_multi: mixed query from human annotation
## Requirements for evaluation
* python3.8
* pytorch=1.10.1+cu113
* pytorch-lightning=1.5.10
* transformers=4.6.1
* timm=0.4.12
* opencv-python
* einops
* wandb
```bash=
conda create --name multimodal python=3.8 pandas numpy 
conda activate multimodal
pip install -r requirements.txt
wandb login
```

## Preprocess
1. Download Mr. Right dataset.
2. Extract the Mr_Right.tar.gz to data/ directoy.
3. Download images and create path for each image (Be sure that your storage is more than 1.5TB)
4. Add image path to your json files: {id:0, ......,"doc_image": "xxx.jpg"}, including multimodal_documents.json, multimodal_train_pairs.json, and multimodal_finetune_pairs.json
```bash=
bash download_dataset.sh
```

## Model Checkpoint
* ALBEF
* METER
* ViLT
```bash=
bash ./checkpoints/download_checkpoints.sh
```
* Download pretrain models from their repositories if you want to train by yourself. Remember to set the path in config file
    * ALBEF: <a href="https://github.com/salesforce/ALBEF"> ALBEF_4M.pth</a>
    * ViLT: <a href="https://github.com/dandelin/ViLT"> vilt_200k_mlm_itm.ckpt</a>
    * METER: <a href="https://github.com/zdou0830/METER"> meter_clip16_288_roberta_pretrain.ckpt</a>

## Edit Configs
* In configs/ALBEF.yaml, configs/METER.yaml, or ViLT.yaml set the paths for the json files and the image path.
* In our task, there are large numbers of documents. To improve the efficiency of validation per training epoch, we suggest that you should split small numbers of doucments with multimodal_val_queries.json. Remember to reset the document id and set the document path in config file.
```bash=
python extract_multimodal_val.py --mul_doc multimodal_documents.json \
--mul_val multimodal_val_queries.json \ 
--val_amount 10000 \ 
--output multimodal_val_documents.json
```

## Fine-tune Multimodal model
* embeds_feats: average embedding or compute cls embedding
* pl_checkpoint: resume from checkpoint

```bash=
CUDA_VISIBLE_DEVICES=0 python main.py \
--num_gpus [number of gpus] \
--num_workers [number of workers] \
--wandb_task_name [Name of task] \
--batch_size 16 \ 
--pretrain [ALBEF | ViLT | METER] \ 
--embeds_feats [avg | cls] \ 
--pl_checkpoint [path for resumed model] \
--save_checkpoint [path for saving checkpoints] \
--neg_matching
--ctx_prediction
--re_ranking
```
## Evaluate 
```bash=
python main.py --task train \
--config configs/model_retrieval.yaml \
--batch_size 64 \ 
--num_workers 32 \ 
--shuffle True \ 
--checkpoint [Pretrained checkpoint] \
```

## Benchmark
![Mr. Right Benchmark](https://github.com/hsiehjackson/Mr.Right/blob/main/benchmark.png?raw=true)



## License
This data is available under the [Creative Commons Attribution Share Alike 4.0](LICENSE) license.

## Contact
For any questions please contact r09944010@ntu.edu.tw or c2hsieh@ucsd.edu 