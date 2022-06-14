import random
import torch
import os
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomResizedCrop, RandomHorizontalFlip
from pytorch_lightning import LightningDataModule
from data.utils import pre_caption, RandomAugment
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TextToMultiDataset(Dataset):
    def __init__(self, args, configs, data, task, tokenizer):
        self.data = data
        self.image_size = configs.image_res
        self.image_root = configs["image_root"]
        self.task = task
        self.tokenizer = tokenizer
        self.configs = configs
        self.args = args
        self.q_max_len = configs.query_length
        self.d_max_len = configs.text_length

        normalize = Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))
        if task == "train":
            self.transform = Compose([
                        RandomResizedCrop(self.image_size,
                                    scale=(0.5, 1.),
                                    interpolation=Image.BICUBIC),
                        RandomHorizontalFlip(),
                        RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']), 
                        ToTensor(),
                        normalize,
                        ])
        elif task == "val_queries":
            self.data = self.data
        elif task == 'test_queries':
            self.data = self.data
        elif task == "docs":
            self.data = self.data
            self.transform = Compose([
                        Resize((self.image_size,self.image_size),
                                    interpolation=Image.BICUBIC),
                        ToTensor(),
                        normalize,
                        ])   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):   
        result = {}
        data = self.data[index]

        if self.task == "train":
            # query str
            result['query_text_str'] = data["txt_query_str"]
            result['query_image_str'] = data["img_query_str"]
            
            r = random.random()  
            if r < 0.33:
                query_str = data["txt_query_str"]
            elif 0.33 < r < 0.66:
                query_str = data["img_query_str"]
            elif 0.66< r < 0.83:
                query_str = data["txt_query_str"] + data["img_query_str"]
            else:
                query_str = data["img_query_str"] + data["txt_query_str"]
            
            image = data["doc_image"]
            image = image.replace("./","")
            image_path = os.path.join(self.image_root,image)        
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)

            result['query_str'] = query_str
            result['doc_id'] = data["doc_id"]
            result['doc_str'] = data["doc_text_str"]
            result['doc_image'] = image
            result['image_path'] = image_path

        elif self.task == 'val_queries':
            result['img_query_str'] = data['img_query_str']
            result['txt_query_str'] = data['txt_query_str']
            result['multi_query_str'] = data['multi_query_str']
            result['doc_id'] = data["doc_id"]

        elif self.task == 'test_queries':
            result['img_query_str'] = data['img_query_str']
            result['txt_query_str'] = data['txt_query_str']
            result['multi_query_str'] = data['multi_query_str']
            result['doc_id'] = data["doc_id"]

        elif self.task == 'docs':
            image = data["doc_image"]
            image = image.replace("./","")
            image_path = os.path.join(self.image_root,image)        
            image = Image.open(image_path).convert('RGB')   
            image = self.transform(image)
            result['doc_str'] = data["doc_text_str"]
            result['doc_image'] = image
            result['image_path'] = image_path
            result['doc_id'] = data["doc_id"]

        return result


    def collate_fn(self, batch):
        if self.task == 'test_queries' or self.task == 'val_queries':
            batch_dict = {
                "doc_id": torch.tensor([b["doc_id"] for b in batch]).long(),
                "img_query_str": [b["img_query_str"] for b in batch],
                "txt_query_str": [b["txt_query_str"] for b in batch],
                "multi_query_str": [b["multi_query_str"] for b in batch],
            }
            img_query_str_tensor = self.tokenizer(
                                text=[b["img_query_str"] for b in batch], 
                                max_length=self.q_max_len, 
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
            txt_query_str_tensor = self.tokenizer(
                                text=[b["txt_query_str"] for b in batch], 
                                max_length=self.q_max_len, 
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
            multi_query_str_tensor = self.tokenizer(
                                text=[b["multi_query_str"] for b in batch], 
                                max_length=self.q_max_len, 
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")
            batch_dict["img_query_str_tensor"] = img_query_str_tensor
            batch_dict["txt_query_str_tensor"] = txt_query_str_tensor
            batch_dict["multi_query_str_tensor"] = multi_query_str_tensor
            return batch_dict

        if self.task == 'docs':
            batch_dict = {
                "doc_id": torch.tensor([b["doc_id"] for b in batch]).long(),
                "image_path":  [b["image_path"] for b in batch],
                "doc_str" : [b["doc_str"] for b in batch],
                "doc_image_tensor": torch.stack([b["doc_image"] for b in batch]),
            }
            doc_str_tensor   = self.tokenizer(
                                    text=[b["doc_str"] for b in batch], 
                                    max_length=self.d_max_len, 
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            batch_dict["doc_str_tensor"] = doc_str_tensor
            return batch_dict

        
        
        
        batch_dict = {
            "doc_id": torch.tensor([b["doc_id"] for b in batch]).long(),
            "query_str": [b["query_str"] for b in batch],
            "doc_str" : [b["doc_str"] for b in batch],
            "doc_image_tensor": torch.stack([b["doc_image"] for b in batch]),
        }
        query_str_tensor = self.tokenizer(
                                text=[b["query_str"] for b in batch], 
                                max_length=self.q_max_len, 
                                padding="longest" if self.task == "train" else "max_length",
                                # padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        batch_dict["query_str_tensor"] = query_str_tensor

        doc_str_tensor   = self.tokenizer(
                                text=[b["doc_str"] for b in batch], 
                                max_length=self.d_max_len, 
                                padding="longest" if self.task == "train" else "max_length",
                                # padding="max_length",
                                truncation=True,
                                return_tensors="pt")
        batch_dict["doc_str_tensor"] = doc_str_tensor

        query_str_tensor_total = query_str_tensor
        if self.task == "train":
            query_str_tensor_total = self.tokenizer(
                                    text=[b["query_text_str"] + ' ' + b["query_image_str"] for b in batch], 
                                    max_length=self.q_max_len, 
                                    padding="longest",
                                    # padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")


        
        if self.args.ctx_prediction:
            context_labels = torch.zeros(len(batch), self.tokenizer.vocab_size)
            pad_id = self.tokenizer.pad_token_id
            sep_id = self.tokenizer.sep_token_id
            cls_id = self.tokenizer.cls_token_id	
            context_labels[torch.arange(len(batch)).unsqueeze(1), query_str_tensor_total['input_ids']] = 1
            context_labels[torch.arange(len(batch)).unsqueeze(1), doc_str_tensor['input_ids']] = 1
            context_labels[:, pad_id] = 0
            context_labels[:, sep_id] = 0
            context_labels[:, cls_id] = 0
            batch_dict["context_labels"] = context_labels

        return batch_dict
                


class TextToMultiDataModule(LightningDataModule):
    def __init__(self,args,configs,tokenizer):
        super().__init__()
        self.args = args
        self.shuffle = args.shuffle
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.configs = configs
        self.max_len = configs.text_length
        self.tokenizer = tokenizer


    def prepare_data(self,train=None,val=None,test=None,document=None):
        # called only on 1 GPU

        def load_data(config, mode):
            datas = self.prepare_text2multi_data(config,mode)
            return datas
        
        if train is not None: self.train_datas = load_data(train, 'train')
        if val is not None: self.val_datas = load_data(val, 'val')
        if test is not None: self.test_datas = load_data(test, 'test')
        if document is not None: self.documents = load_data(document,'doc')

    def setup(self):
        if hasattr(self, 'train_datas'):
            self.train_dataset = TextToMultiDataset(self.args, self.configs, self.train_datas,"train", self.tokenizer)
        if hasattr(self, 'val_datas'):
            self.val_queries_dataset = TextToMultiDataset(self.args, self.configs, self.val_datas, "val_queries", self.tokenizer)
            self.val_docs_dataset = TextToMultiDataset(self.args, self.configs, self.documents, "docs", self.tokenizer)
        if hasattr(self, 'test_datas'):
            self.test_queries_dataset = TextToMultiDataset(self.args, self.configs, self.test_datas, "test_queries", self.tokenizer)
            self.test_docs_dataset  = TextToMultiDataset(self.args, self.configs, self.documents, "docs", self.tokenizer)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True,drop_last=True, collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return [
            DataLoader(self.val_queries_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.val_queries_dataset.collate_fn),
            DataLoader(self.val_docs_dataset,    batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.val_docs_dataset.collate_fn),
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_queries_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.test_queries_dataset.collate_fn),
            DataLoader(self.test_docs_dataset,    batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.test_docs_dataset.collate_fn),
        ]
    
    def prepare_text2multi_data(self,files,task="train"):
        print("\nReading json files")
        data = []
        for f in files:
            print(f"File: {f}",end="\r")
            data += json.load(open(f,'r'))
        result = []
        for idx,pairs in enumerate(data):
            if task == "train":
                result.append({
                    "img_query_str": pre_caption(pairs["query_img"],self.max_len),
                    "txt_query_str":  pre_caption(pairs["query_text"],self.max_len),
                    "doc_text_str": pre_caption(pairs["doc_text"], self.max_len),
                    "doc_image":  pairs["doc_image"],
                    "doc_id": idx,
                })
                
            if task == "val": 
                result.append({
                    "multi_query_str": pre_caption(pairs["query_multi"],self.max_len),
                    "img_query_str": pre_caption(pairs["query_img"],self.max_len),
                    "txt_query_str":  pre_caption(pairs["query_text"],self.max_len),
                    "doc_id": pairs["id"],
                })         
                
            if task == "test":
                result.append({
                    "img_query_str": pre_caption(pairs["query_img"],self.max_len),
                    "txt_query_str": pre_caption(pairs["query_text"],self.max_len),
                    "multi_query_str": pre_caption(pairs["query_multi"],self.max_len),
                    "doc_id": pairs["id"],
                })

            if task == "doc":
                result.append({
                    "doc_text_str": pre_caption(pairs["doc_text"], self.max_len),
                    "doc_image": pairs["doc_image"],
                    "doc_id": pairs["id"],
                })
        return result
        




