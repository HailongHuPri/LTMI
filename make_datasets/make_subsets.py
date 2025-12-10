import os
import h5py
import argparse
import numpy as np
import torchvision


'''
Prepare the dataset (balanced or long-tailed) for model training.
'''


def get_img_num_per_cls(cls_num, imb_type, imb_factor, dataset_len):
    img_max = (dataset_len / cls_num)  
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    else:
        raise ValueError(f"Invalid imbalance type: {imb_type}")
    return img_num_per_cls


def main(args):
    
    seed = args.seed  
    np.random.seed(seed)

    num_subsets = 64  
    appearances_per_sample = num_subsets // 2  

    if args.dataname == 'cifar10':
        cls_num = 10  
        trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True)

    
    img_num_per_cls = get_img_num_per_cls(cls_num, 'exp', args.im_ratio, len(trainset))

    if args.is_longtail == 'F':
        if args.dataname == 'cifar10':
            avg_num_per_cls =1400
            img_num_per_cls = [avg_num_per_cls] * cls_num
    
    class_indices = [[] for _ in range(cls_num)]
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)
    

    subdataset_indices = []
    for cls_idx, num in enumerate(img_num_per_cls):
        indices = class_indices[cls_idx]
        np.random.shuffle(indices)
        subdataset_indices.extend(indices[:num])
    
    subdataset_labels = [trainset[idx][1] for idx in subdataset_indices]

    
    if not os.path.exists(args.subsetdir):
        os.makedirs(args.subsetdir)
    if args.is_longtail == 'T':
        file_path = args.subsetdir + 'subsetsIdx_longtailed_all.h5py'
    else:
        file_path = args.subsetdir + 'subsetsIdx_balanced_all.h5py'
    with h5py.File(file_path, 'w') as h5file:
        h5file.create_dataset('subsetsIdx_indices', data=subdataset_indices)
        h5file.create_dataset('subsetsIdx_labels', data=subdataset_labels)
        

    class_indices_sub = [[] for _ in range(cls_num)]
    for idx, label in zip(subdataset_indices, subdataset_labels):
        class_indices_sub[label].append(idx)


    subsets = [[] for _ in range(num_subsets)]

    for class_idx in range(cls_num):
        np.random.shuffle(class_indices_sub[class_idx])

        for sample_idx in class_indices_sub[class_idx]:
            chosen_subsets = np.random.choice(num_subsets, appearances_per_sample, replace=False)
            for subset in chosen_subsets:
                subsets[subset].append(sample_idx)

    if not os.path.exists(args.subsetdir):
        os.makedirs(args.subsetdir)
    if args.is_longtail == 'T':
        file_path = args.subsetdir + 'subsetsIdx_longtailed_train.h5py'
    else:
        file_path = args.subsetdir + 'subsetsIdx_balanced_train.h5py'

    with h5py.File(file_path, 'w') as h5file:
        for i, lst in enumerate(subsets):
            h5file.create_dataset(f'subset_{i}', data=lst)
    print('finish making subsets.')




def parse_args():
    parser = argparse.ArgumentParser(description="Make subsets")
    parser.add_argument('--datadir', type=str, default='./data', help='Directory to download the dataset')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--subsetdir', type=str, default='./', help='Path to save the subsets')
    parser.add_argument('--is_longtail', type=str, default='T', help='Whether to use long-tailed distribution')

    parser.add_argument('--dataname', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--im_ratio', type=float, default=0.002, help='Imbalance ratio = 1/imbalanced factor')
    

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)