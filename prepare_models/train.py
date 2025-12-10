import os
import argparse

import h5py
import json
import torch
import models
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


'''
Prepare the model for training and save the predictions for following membership inference.
'''

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)




def train(net, optimizer, criterion, trainloader, testloader, 
          scheduler, num_epochs, device, resultdir,num_classes):
    best_accuracy = 20.0
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()                
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1

            if i % 100 == 99:  
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        scheduler.step()
        training_accuracy = 100 * correct / total
        test_accuracy = test(net, testloader, device, num_classes, is_print=False)
        best_accuracy = save_best_model(net, test_accuracy, best_accuracy, resultdir)
        print(f'Epoch {epoch+1} Summary: Train Accuracy: {training_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        
    print(f'Training completed.')

def test(net, testloader, device, num_classes, is_print=False, is_train=False, resultdir=None):
    net.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1

    accuracy = 100 * correct / total

    if is_print == True:
        if is_train == True:
            dataLabel = 'train'
        else:
            dataLabel = 'test'
        for i in range(num_classes):
            print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
        # save the mertics to a json file
        with open(os.path.join(resultdir, f'metrics_{dataLabel}.json'), 'w') as f:
            metrics = {
                'data_label': dataLabel,
                'class_accuracy': [100 * class_correct[i] / class_total[i] for i in range(num_classes)]
            }
            json.dump(metrics, f)

    return accuracy

def save_best_model(net, accuracy, best_accuracy, resultdir):
    model_file = os.path.join(resultdir, 'best_model.pth')
    if accuracy > best_accuracy:
        torch.save(net.state_dict(), model_file)
        return accuracy
    return best_accuracy

def compute_save_predictions(net, members_loader, nonmbers_loader, device, resultdir):
    net.eval()
    pred_logits = []
    true_labels = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(members_loader):
            true_labels.append(targets)
            inputs = inputs.to(device)
            outputs = net(inputs)

            pred_logits.append(outputs.cpu().unsqueeze(1))
            if i % 10000 == 0:
                print(f'Batch [{i+1}/{len(members_loader)}]')
        pred_logits = torch.cat(pred_logits, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
    pred_logits_np_memb = pred_logits.numpy().astype(np.float64)
    true_labels_np_memb = true_labels.numpy().astype(np.float64)  


    pred_logits = []
    true_labels = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(nonmbers_loader):
            true_labels.append(targets)
            inputs = inputs.to(device)
            outputs = net(inputs)
            pred_logits.append(outputs.cpu().unsqueeze(1))

            if i % 10000 == 0:
                print(f'Batch [{i+1}/{len(nonmbers_loader)}]')
        pred_logits = torch.cat(pred_logits, dim=0)
        true_labels = torch.cat(true_labels, dim=0)

    pred_logits_np_nonmemb = pred_logits.numpy().astype(np.float64)
    true_labels_np_nonmemb = true_labels.numpy().astype(np.float64)  

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    output_path_file = os.path.join(resultdir, 'logits.h5py')
    
    with h5py.File(output_path_file, 'w') as h5f:
        h5f.create_dataset('pred_logits_memb', data=pred_logits_np_memb)
        h5f.create_dataset('true_labels_memb', data=true_labels_np_memb)    
        h5f.create_dataset('pred_logits_nonmemb', data=pred_logits_np_nonmemb)
        h5f.create_dataset('true_labels_nonmemb', data=true_labels_np_nonmemb)   

def main():

    args = parse_args()
    np.random.seed(args.seed)
    
    modelIdx = args.modelIdx
    resultdir = os.path.join(args.result_base_dir, "shadow", f"model_{modelIdx:03d}")
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    
    if args.dataname == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataname}")


    with h5py.File(args.subsetsIdxFullPath , 'r') as h5file:
        subdataset_indices = list(h5file['subsetsIdx_indices'][:])

    with h5py.File(args.subsetsIdxTrainPath, 'r') as h5file:
        subkey = f'subset_{modelIdx}'
        subsetsIdx= list(h5file[subkey][:])
        print(f"Subset {subkey} has {len(subsetsIdx)} samples.")


    trainset_subset = Subset(trainset, subsetsIdx)
    trainloader_subset = DataLoader(trainset_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    

    nonmbersIdx = list(set(subdataset_indices)-set(subsetsIdx))
    nonmbers_subset = Subset(trainset, nonmbersIdx)
    nonmbers_loader = DataLoader(nonmbers_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = models.WideResNet(
        in_channels=3,
        depth=16,
        widen_factor=4,
        num_classes=num_classes,
        use_group_norm=False,
        device=device,
        dtype=torch.float32,
    )    
         
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        net = nn.DataParallel(net)
        
    net = net.to(device)

    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    num_steps_per_epoch = 0
    for _ in trainloader_subset:
        num_steps_per_epoch += 1

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / num_steps_per_epoch,
                end_factor=1.0,
                total_iters=num_steps_per_epoch,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch * num_steps_per_epoch for epoch in (60, 120, 160)], gamma=0.2
            ),
        ],
        milestones=[1 * num_steps_per_epoch],
    )


    train(net, optimizer, criterion, trainloader_subset, testloader,
          scheduler, args.num_epochs, device, resultdir, num_classes)
    

    net.load_state_dict(torch.load(os.path.join(resultdir, 'best_model.pth')))
    print('Training accuracy on each class:')
    train_accuracy = test(net, trainloader_subset, device, num_classes, is_print=True, 
                          is_train=True,resultdir=resultdir)
    print('Test accuracy on each class:')
    test_accuracy = test(net, testloader, device, num_classes,
                         is_print=True, is_train=False, resultdir=resultdir)


    members_loader = DataLoader(trainset_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    compute_save_predictions(net, members_loader, nonmbers_loader, device, resultdir)



def parse_args():
    parser = argparse.ArgumentParser(description="Training a model")
    parser.add_argument('--datadir', type=str, default='./data', help='Directory to download the dataset')
    parser.add_argument('--dataname', type=str, default='cifar10', help='Name of the dataset')
    
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--result_base_dir', type=str, default='./', help='Directory to save the results')
    
    parser.add_argument('--modelIdx', type=int, default=0, help='Model index')
    parser.add_argument('--subsetsIdxTrainPath', type=str, default='./', help='Path to the subset index file')
    parser.add_argument('--subsetsIdxFullPath', type=str, default='./', help='Path to the subset index file')
   
    return parser.parse_args()

if __name__ == "__main__":
    main()