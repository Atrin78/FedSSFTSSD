"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from typing import Tuple, Optional, List, Dict
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as func
from torch.optim import SGD
import matplotlib.pyplot as plt


def pgd_attack(model, data, labels, loss_fun, device, eps=0.05, alpha=0.003125, iters=40):
    data = data.to(device).float()
    labels = labels.to(device).long()

    ori_data = data.clone().detach().to(device).float()

    for i in range(iters):
        data.requires_grad = True
        outputs,_ = model(data)

        model.zero_grad()
        cost = loss_fun(outputs, labels).to(device)
        cost.backward()

        adv_data = data - alpha * data.grad.sign()
        eta = torch.clamp(adv_data - ori_data, min=-eps, max=eps)
        #       data = torch.clamp(ori_data + eta, min=0, max=1).detach_()
        data = ori_data + eta
        data = data.detach_()

    return data.to(torch.device("cpu"))

def labels_to_one_hot(labels, num_class, device):
    # convert labels to one-hot
    labels_one_hot = torch.FloatTensor(labels.shape[0], num_class).to(device)
    labels_one_hot.zero_()
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot

def src_img_synth_admm(gen_loader, src_model, args):

  #  gen_folder = 'gen_data_admm/'

  #  data_list_file = args.root
  #  data_list_file = data_list_file.replace('data/', gen_folder)
  #  data_list_file += '/image_list/'+args.source+'2'+args.target+'.txt'
    
  #  dir = os.path.dirname(data_list_file)
  #  if not os.path.exists(dir):
  #      os.makedirs(dir)

    # initialize
  #  for batch_idx, images_t in enumerate(data_loader):

  #      if batch_idx == 0 and os.path.exists(data_list_file):
  #          os.remove(data_list_file)

  #      images_t = images_t.to(device)
        # get pseudo labels
  #      y_t = src_model(images_t)
  #      plabel_t = y_t.argmax(dim=1)

   #     save_src_imgs(images_t.cpu(), plabel_t, path, gen_folder, data_list_file, args)

 #   genset = DataSetGen(data_list_file)
 #   genset_path = DataSetPath(genset)
 #   gen_loader = DataLoader(genset_path, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    LAMB = torch.zeros_like(src_model.head.weight.data).to(device)
    gen_dataset = None
    gen_labels = None
    for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
        if batch_idx == 100:
            break
        images_s = images_s.to(device)
        y_s,_ = src_model(images_s)
        labels_s = y_s.argmax(dim=1)
        if gen_dataset == None:
            gen_dataset = images_s
            gen_labels = labels_s
        else:
            gen_dataset = torch.cat((gen_dataset, images_s), 0)
            gen_labels = torch.cat((gen_labels, labels_s), 0)

    for i in range(args.iters_admm):

        print(f'admm iter: {i}/{args.iters_admm}')

        # step1: update imgs
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            if batch_idx==0 and i==0:
                for j in range(10):
                    print('hi')
                    print(gen_labels[j])
                    plt.imshow(np.moveaxis(gen_dataset[j].cpu().detach().numpy(), 0, -1))
                    plt.savefig("im"+str(j))
            if batch_idx == 100:
                break

    #        images_s = images_s.to(device)
    #        labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, 10, device)

            # init src img
            images_s.requires_grad_()
            optimizer_s = SGD([images_s], args.lr_img, momentum=args.momentum_img)
            
            for iter_i in range(args.iters_img):
                # if batch_idx == 0:
                #     for i in range(10):
                #         plt.imshow(np.moveaxis(images_s[i].cpu().detach().numpy(), 0, -1))
                #         plt.savefig("step" + str(iter_i)+'-' + str(i))
                y_s, f_s = src_model(torch.clip(images_s, 0.0, 255.0))
                loss = func.cross_entropy(y_s, labels_s)
                p_s = func.softmax(y_s, dim=1)
                grad_matrix = (p_s - plabel_onehot).t() @ f_s / p_s.size(0)
                new_matrix = grad_matrix + args.param_gamma * src_model.head.weight.data
                grad_loss = torch.norm(new_matrix, p='fro') ** 2
                loss += grad_loss * args.param_admm_rho / 2
                loss += torch.trace(LAMB.t() @ new_matrix)
                
                optimizer_s.zero_grad()
                loss.backward()
                # print('grad')
                # print(images_s.grad)
                optimizer_s.step()


            # update src imgs
            gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch] = torch.clip(images_s, 0.0, 255.0)
       #     for img, path in zip(images_s.detach_().cpu(), paths):
       #         torch.save(img.clone(), path)

        # step2: update LAMB
        grad_matrix = torch.zeros_like(LAMB).to(device)
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            if batch_idx == 100:
                break
       #     images_s = images_s.to(device)
       #     labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*args.batch:(batch_idx+1)*args.batch].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, 10, device)

            y_s, f_s = src_model(images_s)
            p_s = func.softmax(y_s, dim=1)
            grad_matrix += (p_s - plabel_onehot).t() @ f_s

        new_matrix = grad_matrix / len(gen_dataset) + args.param_gamma * src_model.head.weight.data
        LAMB += new_matrix * args.param_admm_rho

    return gen_dataset, gen_labels



class BackBone(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
   #     self.fc2 = nn.Linear(2048, 512)
   #     self.bn5 = nn.BatchNorm1d(512)
   #     self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

    #    x = self.fc2(x)
    #    x = self.bn5(x)
    #    x = func.relu(x)

    #    x = self.fc3(x)
        return x

class ImageClassifier(nn.Module):

    def __init__(self, num_classes: int = 10, bottleneck_dim: Optional[int] = 512,**kwargs):
        super(ImageClassifier, self).__init__()
        self.backbone = BackBone()
        self.num_classes = num_classes
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self._features_dim = bottleneck_dim
        self.head = nn.Linear(self._features_dim, num_classes)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        x = self.backbone(x)
        f = self.bottleneck(x)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params

def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)

    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)

    # Synth Digits
    # train changed to test
    synth_trainset     = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=False,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
    synth = torch.utils.data.ConcatDataset([synth_trainset, synth_testset])    

    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
 #   synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
 #   synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    synth_loader = torch.utils.data.DataLoader(synth, batch_size=args.batch,  shuffle=True)
    print(len(synth_loader.dataset))
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, mnistm_test_loader, synth_loader]

    return train_loaders, test_loaders

def train(model, train_loader, optimizer, loss_fun, client_num, device, robust_training):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        if robust_training:
            x = pgd_attack(model, x, y, loss_fun, device)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,_ = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output,_ = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all/len(train_iter), correct/num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output,_ = model(data)
        
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()
    
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 256, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=10, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedbn', help='fedavg | fedprox | fedbn')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--synth_method', type = str, default='admm', help='admm | ce')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--momentum_img', default=0.9, type=float, metavar='M',
                        help='momentum of img optimizer')
    parser.add_argument('--iters_img', default=10, type=int, metavar='N',
                        help='number of total inner epochs to run')
    parser.add_argument('--param_gamma', default=0.01, type=float)
    parser.add_argument('--param_admm_rho', default=0.01, type=float)
    parser.add_argument('--iters_admm', default=10, type=int)
    parser.add_argument('--lr_img', default=100., type=float)
    parser.add_argument('--begin_generation', default=1, type=int)
    args = parser.parse_args()

    exp_folder = 'federated_digits'

    args.save_path = os.path.join(args.save_path, exp_folder)
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))
   
   
    server_model = ImageClassifier(10,512).to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'MNIST-M', 'SynthDigits']
    
    # federated setting
    client_num = len(datasets)-1
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(models[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower()=='fedbn':
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                models[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        # freezing server model
        for param in server_model.parameters():
            param.requires_grad = False
        server_model.eval()

        # if a_iter >= 40:
        #     if args.synth_method == 'ce':
        #         pass
        #     elif args.synth_method == 'admm':
        #         vir_dataset, vir_labels = src_img_synth_admm(test_loaders[client_num], server_model, args)
        #
        # if a_iter==40:
        #     for i in range(10):
        #         plt.imshow(np.moveaxis(vir_dataset[i].cpu().detach().numpy(), 0, -1))
        #         plt.savefig("vir"+str(i))
        

        optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(client_num)]


        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log: logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters)) 
            
            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device, False)
                else:
                    if a_iter == args.begin_generation:
                        train(model, train_loader, optimizer, loss_fun, client_num, device, True)
                    else:
                        train(model, train_loader, optimizer, loss_fun, client_num, device, False)

        for param in models[0].parameters():
            param.requires_grad = False
        models[0].eval()
        if a_iter >= args.begin_generation:
            vir_dataset, vir_labels = src_img_synth_admm(test_loaders[client_num], models[0], args)
        if a_iter== args.begin_generation:
            for i in range(10):
                plt.imshow(np.moveaxis(vir_dataset[i].cpu().detach().numpy(), 0, -1))
                plt.savefig("vir"+str(i))
        for param in models[0].parameters():
            param.requires_grad = True
        models[0].eval()
        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)
        
        # report after aggregation
        for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss, train_acc = test(model, train_loader, loss_fun, device) 
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\

        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            if test_idx==client_num:
                break
            test_loss, test_acc = test(models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.mode.lower() == 'fedbn':
        torch.save({
            'model_0': models[0].state_dict(),
            'model_1': models[1].state_dict(),
            'model_2': models[2].state_dict(),
            'model_3': models[3].state_dict(),
            'model_4': models[4].state_dict(),
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)
    else:
        torch.save({
            'server_model': server_model.state_dict(),
        }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()






