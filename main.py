import os
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import scipy.io as io
from torch_geometric.data import Data
from imports.AGEDataset import AGEDataset
from torch_geometric.loader import DataLoader
from net.model import SurfGNN
from imports.utils import train_val_test_split
import csv
import datetime
import pandas as pd


torch.manual_seed(123)
EPS = 1e-10
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

parser.add_argument('--times', type=int, default=1, help='times of running')
parser.add_argument('--batchSize', type=int, default=12, help='size of the batches')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=50, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
# hyperparameters
parser.add_argument('--lamb1', type=float, default=1, help='feature loss regularization')
parser.add_argument('--lamb2', type=float, default=1, help='unit loss regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='score loss regularization')
parser.add_argument('--ratio', type=float, default=0.7, help='pooling ratio for each Nodal_GPool')
# model
parser.add_argument('--num_nodes', type=str, default=20484, help='num of ROIs/nodes input into model/TSL')
parser.add_argument('--num_TSL', type=int, default=3, help='num of TSL (influence num of ROIs/nodes input into RSL)')
parser.add_argument('--input_RSL', type=int, default=324, help='num of ROIs/nodes input into RSL')
parser.add_argument('--indim', type=int, default=5, help='feature dim / num of channels')
parser.add_argument('--nclass', type=int, default=1, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
# root
parser.add_argument('--psave_path', type=str, default='/opt/data/private/J_surfacebased/braindata_ukbb/model/parameter/',
                    help='path to save model parameter')
parser.add_argument('--rsave_path', type=str, default='/opt/data/private/J_surfacebased/braindata_ukbb/model/result/',
                    help='path to save model result')
parser.add_argument('--dataroot', type=str, default='/opt/data/private/J_surfacebased/braindata_ukbb/',
                    help='root directory of the dataset')


opt = parser.parse_args()

input_RSL = opt.num_nodes
for i in range(opt.num_TSL):
    input_RSL = int((input_RSL + 12) / 4)
assert input_RSL >= 324
opt.input_RSL = input_RSL

if not os.path.exists(opt.psave_path):
    os.makedirs(opt.psave_path)
if not os.path.exists(opt.rsave_path):
    os.makedirs(opt.rsave_path)


############# loss ###############
def RMSE_loss(predict, data_y):
    return torch.sqrt(torch.mean((predict - data_y) ** 2))


def MAE_loss(predict, data_y):
    return torch.mean(abs(predict - data_y))


def unit_loss(w):
    return (torch.norm(w.squeeze(), p=2, dim=-1) - 1) ** 2


def score_loss(s):
    # s = s.view(s.shape[0], -1)
    ratio = 1 - 0.5  # 下限0.3
    s = abs(s).sort(dim=1).values

    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() - torch.log(
        1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    return res


def loss_concat(y, output, out_feature, w_pool, score):
    loss_m = MAE_loss(output.squeeze(dim=1), y)
    loss_c = F.mse_loss(output.squeeze(dim=1), y)
    loss_feature = []
    for ff in range(out_feature.size(-1)):
        loss_feature.append(F.mse_loss(out_feature[:, ff], y))
    loss_unit = unit_loss(w_pool)
    loss_score = score_loss(score)

    return loss_m, loss_c, loss_feature, loss_unit, loss_score


############## train function #############
def train(epoch):
    print("train----------------")
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()

    score_list = []
    loss_all = 0
    loss_MAE = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, out_feature, w_pool, score, _, _, _ = model(data.x, data.batch)

        loss_m, loss_c, loss_feature, loss_unit, loss_score = loss_concat(data.y, output,
                                                                          out_feature, w_pool, score)

        loss = loss_c + opt.lamb1 * sum(loss_feature) + opt.lamb2 * loss_unit + opt.lamb3 * loss_score

        loss_MAE += loss_m.item() * data.num_graphs
        score_list.extend(score.view(-1).detach().cpu().numpy())
        writer.add_scalar('train/classification_mse_loss', loss_c, epoch * len(train_loader) + step)
        writer.add_scalar('train/mae', loss_m, epoch * len(train_loader) + step)
        for ff in range(len(loss_feature)):
            writer.add_scalar('train/mse_feature' + str(ff + 1), loss_feature[ff], epoch * len(train_loader) + step)
        writer.add_scalar('train/score', loss_score, epoch * len(train_loader) + step)
        writer.add_scalar('train/unit', loss_unit, epoch * len(train_loader) + step)

        step = step + 1
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()

    score_arr = np.array(score_list)
    writer.add_histogram('Hist/hist_score', score_arr, epoch)

    loss_all_val = 0
    loss_MAE_val = 0
    model.eval()
    for data in val_loader:
        data = data.to(device)

        output, out_feature, w_pool, score, _, _, _ = model(data.x, data.batch)

        loss_m, loss_c, loss_feature, loss_unit, loss_score = loss_concat(data.y, output,
                                                                          out_feature, w_pool, score)

        loss = loss_c + opt.lamb1 * sum(loss_feature) + opt.lamb2 * loss_unit + opt.lamb3 * loss_score

        loss_MAE_val += loss_m.item() * data.num_graphs
        loss_all_val += loss.item() * data.num_graphs

    return loss_all / len(train_dataset), score_arr, loss_all_val / len(val_dataset), \
           loss_MAE / len(train_dataset), loss_MAE_val / len(val_dataset)


############### test function ####################
def test(best_model, loader):
    print("test------------------")
    model.load_state_dict(best_model)
    model.eval()
    loss_te = 0
    pred_test = []
    true_test = []
    perm_list_1 = []
    perm_list_2 = []
    perm_list_all = []
    score_list = []
    for data in loader:
        data = data.to(device)
        output, out_feature, w_pool, score, perm1, perm2, perm_all = model(data.x, data.batch)

        loss_m, loss_c, loss_feature, loss_unit, loss_score = loss_concat(data.y, output,
                                                                          out_feature, w_pool, score)

        loss = loss_c + opt.lamb1 * sum(loss_feature) + opt.lamb2 * loss_unit + opt.lamb3 * loss_score
        loss_te += loss.item() * data.num_graphs
        pred_test.extend(output.squeeze(1).cpu().detach().numpy())
        label = data.y
        true_test.extend(label.cpu().detach().numpy())

        perm_list_1.extend(perm1.detach().cpu().numpy())
        perm_list_2.extend(perm2.detach().cpu().numpy())
        perm_list_all.extend(perm_all.detach().cpu().numpy())
        score_list.extend(score.detach().cpu().numpy())

    perm_arr_1 = np.array(perm_list_1)
    perm_arr_2 = np.array(perm_list_2)
    perm_arr_all = np.array(perm_list_all)

    score_arr = np.array(score_list)

    return loss_te / len(pred_test), pred_test, true_test, perm_arr_1, perm_arr_2, perm_arr_all, score_arr


################## run ##################
def run(model):

    #### train ####
    best_model_wts = copy.deepcopy(model.state_dict())
    if opt.load_model:
        best_model_wts = torch.load(os.path.join(opt.psave_path, param2save + '.pth'))
        model.load_state_dict(best_model_wts)
    best_loss = 1e10
    loss_li_tr = []
    loss_li_va = []
    for epoch in range(0, opt.n_epochs):
        since = time.time()
        tr_loss, score_arr, val_loss, tr_MAE, val_MAE = train(epoch)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}, tr_MAE: {:.7f}, '
              'val_MAE: {:.7f}'.format(epoch, tr_loss, val_loss, tr_MAE, val_MAE))
        loss_li_tr.append(tr_loss)
        loss_li_va.append(val_loss)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss}, epoch)
        scheduler.step()

        if val_loss < best_loss and epoch > 2:
            print("saving best model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if opt.save_model:
                torch.save(best_model_wts,
                           os.path.join(opt.psave_path, param2save + '.pth'))
    ##### test ####

    loss_test, pred_test, true_test, perm_1, perm_2, perm_all, score = test(best_model_wts, test_loader)
    true_test = np.asarray(true_test)
    pred_test = np.asarray(pred_test)
    assert (true_test.size == pred_test.size)
    test_MAE = (np.absolute(true_test - pred_test)).mean()
    test_pear = np.corrcoef(true_test, pred_test)

    print("===========================")
    print("Test Loss: {:.7f}, Test MAE: {:.7f}, Test pear:{:.7f}".format(loss_test, test_MAE, test_pear[0][1]))

    ##### activation map ####
    pos_score = []
    perm_1 = perm_1.reshape(len(test_dataset), -1, opt.indim)
    perm_2 = perm_2.reshape(len(test_dataset), -1, opt.indim)
    pos_all = perm_all.reshape(len(test_dataset), -1)
    for subj in range(len(test_dataset)):
        b = subj % opt.batchSize
        perm_1[subj] = perm_1[subj] - b * opt.input_RSL
        perm_2[subj] = perm_2[subj] - b * perm_1.shape[1]

        pos_all[subj] = pos_all[subj] - b * perm_2.shape[1] * opt.indim
    for ff in range(opt.indim):
        pos_1 = perm_1[:, :, ff]
        pos_2 = perm_2[:, :, ff]
        pos_arr = np.zeros((len(test_dataset), opt.input_RSL))
        for subj in range(len(test_dataset)):
            for index, item in enumerate(pos_all[subj]):
                line = pos_2.shape[-1]
                if line * (ff + 1) > item >= line * ff:
                    item = item - line * ff
                    pos_arr[subj][pos_1[subj][pos_2[subj][item]]] = score[subj][index]
        pos_score.append(list(pos_arr))

    save_loader = os.path.join(opt.rsave_path, file2save) + '.mat'
    io.savemat(save_loader, {'test_y': true_test, 'pred_y': pred_test, 'test_MAE': test_MAE, 'test_pear': test_pear,
                             'pos_score': np.array(pos_score)})

    ## test with train_dataset
    output = test(best_model_wts, train_loader)
    true_train = np.asarray(output[1])
    pred_train = np.asarray(output[2])
    train_MAE = (np.absolute(true_train - pred_train)).mean()
    train_pear = np.corrcoef(true_train, pred_train)
    print("===========================")
    print("Train Loss: {:.7f}, Train MAE: {:.7f}, Train pear:{:.7f}".format(output[0], train_MAE, train_pear[0][1]))


if __name__ == '__main__':

    dataset = AGEDataset(opt.dataroot, 'Brain_surf', num_nodes=opt.num_nodes)
    dataset.data.y = dataset.data.y.squeeze()
    dataset.data.x[dataset.data.x == float('inf')] = 0
    dataset.data.x = (dataset.data.x - torch.mean(dataset.data.x, 0)) / torch.std(dataset.data.x, 0)


    for fold in range(0, 5):

        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        opt.fold = fold
        print(opt)
        param2save = 'nki_' + str(opt.times) + '_' + str(opt.num_nodes) + '_fold' + str(opt.fold)
        file2save = 'tm-r-nki_' + str(opt.times) + '_' + str(opt.num_nodes) + '_fold' + str(opt.fold)
        writer = SummaryWriter(os.path.join('./log', file2save))

        ########### dataset #############

        tr_index, val_index, te_index = train_val_test_split(fold=opt.fold, num=len(dataset))
        train_dataset = dataset[list(tr_index)]
        val_dataset = dataset[list(val_index)]
        test_dataset = dataset[list(te_index)]

        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

        model = SurfGNN(opt.num_nodes, opt.num_TSL, input_RSL, opt.indim, opt.ratio, opt.nclass).to(device)

        print(model)
        for name, params in model.named_parameters():
            print(name, ':', params.size())
        print(
            "Total number of paramerters in networks is {}  ".format(sum(x.nelement() for x in model.parameters())))

        if opt.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
        elif opt.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.95, weight_decay=opt.weightdecay,
                                        nesterov=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
        run(model)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        file_opt = '/opt/data/private/NKI_age_prediction/model/opt.txt'
        with open(file_opt, 'a') as f:
            f.write(str(nowTime) + '\n')
            f.write(str(file2save) + '\n')
            f.write(str(opt) + '\n')
            f.write('\n')
