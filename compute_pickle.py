import pickle
import os
from argparse import ArgumentParser
from metric import score_v2

def main(args):
    print("Load pickle...")
    path = args.pickle_intput

    multi_doc_path = os.path.join(path,'multimodal_documents.pickle')
    img_embd_path = os.path.join(path,'img_query.pickle')
    txt_embd_path = os.path.join(path,'txt_query.pickle')
    mr_embd_path = os.path.join(path,'multi_query.pickle')
    true_label_path = os.path.join(path,'labels.pickle')


    with open(multi_doc_path, 'rb') as handle:
        all_docs = pickle.load(handle)
    with open(img_embd_path, 'rb') as handle:
        all_img_queries = pickle.load(handle)
    with open(txt_embd_path, 'rb') as handle:
        all_txt_queries = pickle.load(handle)
    with open(mr_embd_path, 'rb') as handle:
        all_multi_queries = pickle.load(handle)
    with open(true_label_path, 'rb') as handle:
        all_queries_doc_ids = pickle.load(handle)

    print(f"Document size: {all_docs.shape}")
    print(f"Text-related query size: {all_txt_queries.shape}")    
    print(f"Image-related query size: {all_img_queries.shape}")
    print(f"Mixed query size: {all_multi_queries.shape}")

    img_sims_matrix = all_img_queries @ all_docs.t()
    img_output_score = score_v2(img_sims_matrix,all_queries_doc_ids)
    print(f"IR score:{img_output_score}")
    del img_sims_matrix

    txt_sims_matrix = all_txt_queries @ all_docs.t()
    txt_output_score = score_v2(txt_sims_matrix,all_queries_doc_ids)
    print(f"TR score:{txt_output_score}")
    del txt_sims_matrix

    multi_sims_matrix = all_multi_queries @ all_docs.t()
    multi_output_score = score_v2(multi_sims_matrix,all_queries_doc_ids)
    print(f"MR score:{multi_output_score}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pickle_intput', default='pickle-albef/')
    args = parser.parse_args()

    print(args)
    main(args)