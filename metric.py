import numpy as np
import torch
import pdb
from torchmetrics.functional import retrieval_recall,retrieval_reciprocal_rank

@torch.no_grad()
def score(scores_t2m, query_doc_id):
    """
    scores_t2m: (q_size, d_size)
    query_doc_id: (q_size)
    """
    ids = query_doc_id.unsqueeze(1)

    top1_i = torch.topk(scores_t2m, k=1, dim=1).indices
    top5_i = torch.topk(scores_t2m, k=5, dim=1).indices
    top10_v, top10_i = torch.topk(scores_t2m, k=10, dim=1)

    r1 = torch.mean(torch.sum(top1_i == ids, dim=1).float()).item() * 100
    r5 = torch.mean(torch.sum(top5_i == ids, dim=1).float()).item() * 100
    r10 = torch.mean(torch.sum(top10_i == ids, dim=1).float()).item() * 100
    rmean = np.mean([r1, r5, r10])

    top10_m = (top10_i==ids)
    mrr10 = np.mean([retrieval_reciprocal_rank(v, m).item() for v, m in zip(top10_v, top10_m)]) *  100 

    r1, r5, r10, mrr10, rmean


    eval_result =  {'r1': r1,
                    'r5': r5,
                    'r10': r10,
                    'mrr10': mrr10,
                    'r_mean': rmean,
                    }
    return eval_result