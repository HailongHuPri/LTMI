import h5py
import argparse
import numpy as np
import scipy.stats

'''
Perform membership inference.
'''

def logit_score(raw_predictions: np.ndarray, labels: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    assert raw_predictions.ndim >= 2 and labels.ndim == 1 and raw_predictions.shape[0] == len(labels)
    raw_predictions = raw_predictions.astype(np.float64)
    labels = labels.astype(int)
    target_predictions = raw_predictions[np.arange(len(labels)), ..., labels]
    raw_predictions[np.arange(len(labels)), ..., labels] = float("-inf")
    logsumexp = np.log(np.sum(np.exp(raw_predictions), axis=-1))  
    return target_predictions - logsumexp


def load_logits(data_path):
    with h5py.File(data_path, 'r') as f:
        pred_logits_memb = f['pred_logits_memb'][:]
        true_labels_memb = f['true_labels_memb'][:]
        pred_logits_nonmemb = f['pred_logits_nonmemb'][:]
        true_labels_nonmemb = f['true_labels_nonmemb'][:]
    
    results = {
        'pred_logits_memb': pred_logits_memb,
        'true_labels_memb': true_labels_memb,
        'pred_logits_nonmemb': pred_logits_nonmemb,
        'true_labels_nonmemb': true_labels_nonmemb
    }
    
    return results

def load_subsetsIdxFull(subsetsIdxFullPath):
    with h5py.File(subsetsIdxFullPath , 'r') as h5file:
        subdataset_indices = list(h5file['subsetsIdx_indices'][:])
    return subdataset_indices

def load_subsetIdxTrain(subsetsIdxTrainPath):
    subsets = {}
    with h5py.File(subsetsIdxTrainPath, 'r') as h5file:
        for key in h5file.keys():
            subset = list(h5file[key][:])
            subsets[key] = subset

    return subsets

def get_subsetIdxNonmemb(subsetsIdxFullPath, subsetsIdxTrainPath):
    with h5py.File(subsetsIdxFullPath , 'r') as h5file:
        subdataset_indices = list(h5file['subsetsIdx_indices'][:])
    
    subsets = {}
    with h5py.File(subsetsIdxTrainPath, 'r') as h5file:
        for key in h5file.keys():
            subset = list(h5file[key][:])
            subsets[key] = subset
    
    all_subsetIdxNonmemb = {}
    for key in subsets.keys():
        all_subsetIdxNonmemb[key] = list(set(subdataset_indices)-set(subsets[key]))

    return all_subsetIdxNonmemb





def lira(member_data, IN_data, OUT_data):
    eps = 1e-30  
    member_data = member_data.astype(np.float64)
    IN_data = np.array(IN_data).astype(np.float64)
    OUT_data = np.array(OUT_data).astype(np.float64)
    
    means_in = np.mean(IN_data, axis=0)
    stds_in = np.std(IN_data, axis=0) + eps

    means_out = np.mean(OUT_data, axis=0)
    stds_out = np.std(OUT_data, axis=0) + eps

    if IN_data.shape[1] > 1:
        log_prs_in = np.mean(scipy.stats.norm.logpdf(member_data, means_in, stds_in),axis=-1)
        log_prs_out = np.mean(scipy.stats.norm.logpdf(member_data, means_out, stds_out),axis=-1)
    else:
        log_prs_in = scipy.stats.norm.logpdf(member_data, means_in, stds_in)
        log_prs_out = scipy.stats.norm.logpdf(member_data, means_out, stds_out)

    
    result_scores = log_prs_in - log_prs_out
    result_scores = result_scores.astype(np.float64)

    return result_scores

def mia_once():
    
    args = parse_args()
    np.random.seed(args.seed)

    all_membIdx = load_subsetIdxTrain(args.subsetsIdxTrainPath)

    all_nonmembIdx = get_subsetIdxNonmemb(args.subsetsIdxFullPath, args.subsetsIdxTrainPath)

    all_indices = load_subsetsIdxFull(args.subsetsIdxFullPath)

    target_membIdx = all_membIdx[f'subset_{args.modelIdx}']
    del all_membIdx[f'subset_{args.modelIdx}']
    shadow_subsetsIdx = all_membIdx

    target_nonmembIdx = all_nonmembIdx[f'subset_{args.modelIdx}']
    del all_nonmembIdx[f'subset_{args.modelIdx}']
    shadow_subsets_nonmembIdx = all_nonmembIdx


    
    logistsPath_base = args.logistsPath
    all_logits_raw = {}

    for i in range(64):
        cur_path = logistsPath_base + f'model_{i:03d}/logits.h5py'
        all_logits_raw[f'subset_{i}'] = load_logits(cur_path)

    
    all_logits={}
    for subset_idx in all_logits_raw.keys():
        pred_logits_memb_raw = all_logits_raw[subset_idx]['pred_logits_memb']
        true_labels_memb_raw = all_logits_raw[subset_idx]['true_labels_memb']
        pred_logits_nonmemb_raw = all_logits_raw[subset_idx]['pred_logits_nonmemb']
        true_labels_nonmemb_raw = all_logits_raw[subset_idx]['true_labels_nonmemb']
        pred_logits_memb = logit_score(pred_logits_memb_raw, true_labels_memb_raw)
        pred_logits_nonmemb = logit_score(pred_logits_nonmemb_raw, true_labels_nonmemb_raw)
        all_logits[subset_idx] = {'pred_logits_memb': pred_logits_memb, \
                                'true_labels_memb': true_labels_memb_raw, \
                                'pred_logits_nonmemb': pred_logits_nonmemb, \
                                'true_labels_nonmemb': true_labels_nonmemb_raw}
        

    membs_scores = []
    membs_class = []
    for idx, memb in enumerate(target_membIdx):
        member_data = all_logits[f'subset_{args.modelIdx}']['pred_logits_memb'][idx] 
        IN_list = []
        IN_listIdx = []
        OUT_list = []
        OUT_listIdx = []
        IN_logits = []
        OUT_logits = []
        for key in shadow_subsetsIdx.keys():
            if memb in shadow_subsetsIdx[key]:
                IN_list.append(key)
                location = shadow_subsetsIdx[key].index(memb)
                IN_listIdx.append(location)
                IN_logits.append(all_logits[key]['pred_logits_memb'][location])
            else:
                OUT_list.append(key)
                location = shadow_subsets_nonmembIdx[key].index(memb)
                OUT_listIdx.append(location)
                OUT_logits.append(all_logits[key]['pred_logits_nonmemb'][location])
        membs_scores.append(lira(member_data, IN_logits, OUT_logits))
        membs_class.append(all_logits[f'subset_{args.modelIdx}']['true_labels_memb'][idx])

    nonmembs_scores = []
    nonmemb_class = []
    for idx, memb in enumerate(target_nonmembIdx):
        member_data = all_logits[f'subset_{args.modelIdx}']['pred_logits_nonmemb'][idx]  # 
        IN_list = []
        IN_listIdx = []
        OUT_list = []
        OUT_listIdx = []
        IN_logits = []
        OUT_logits = []
        for key in shadow_subsetsIdx.keys():
            if memb in shadow_subsetsIdx[key]:
                IN_list.append(key)
                location = shadow_subsetsIdx[key].index(memb)
                IN_listIdx.append(location)
                IN_logits.append(all_logits[key]['pred_logits_memb'][location])
            else:
                OUT_list.append(key)
                location = shadow_subsets_nonmembIdx[key].index(memb)
                OUT_listIdx.append(location)
                OUT_logits.append(all_logits[key]['pred_logits_nonmemb'][location])
        nonmembs_scores.append(lira(member_data, IN_logits, OUT_logits))    
        nonmemb_class.append(all_logits[f'subset_{args.modelIdx}']['true_labels_nonmemb'][idx])
    
    membs_scores = np.array(membs_scores).flatten()
    membs_class = np.array(membs_class).flatten()
    nonmembs_scores = np.array(nonmembs_scores).flatten()
    nonmemb_class = np.array(nonmemb_class).flatten()



    save_path = logistsPath_base + f'model_{args.modelIdx:03d}/mia_data.h5py'
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('membs_scores', data=membs_scores)
        f.create_dataset('membs_class', data=membs_class)
        f.create_dataset('nonmembs_scores', data=nonmembs_scores)
        f.create_dataset('nonmemb_class', data=nonmemb_class)
    print('The mia data is saved.')


def parse_args():
    parser = argparse.ArgumentParser(description="Membership inference")

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument('--subsetsIdxTrainPath', type=str, default='./', help='Path to the subset index file')
    parser.add_argument('--subsetsIdxFullPath', type=str, default='./', help='Path to the subset index file')
    parser.add_argument('--logistsPath', type=str, default='./', help='Path to the logits file')
    parser.add_argument('--modelIdx', type=int, default=0, help='ID of the target model')
    
    return parser.parse_args()

if __name__ == '__main__':
    mia_once()
