from __future__ import print_function
import sys
import torch
import sys, os, gc
import time
import math
import pickle
import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, recall_score, roc_curve, roc_auc_score
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from sklearn.metrics import auc
from torchsummary import  summary
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from hyperopt import hp
#os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class ProtPep_dataset(Dataset):

    def __init__(self, ws, mode):

        self.ws = ws
        self.mode = mode


        input_file = self.mode + '_' + str(self.ws) + '_set.csv'
        label_file = str(self.mode) + '_labels.txt'

        self.image_list = []
        self.label_list = []

        with open(input_file, 'r') as input_f:

            buffer = csv.reader(input_f)
            img = []
            for i, row in enumerate(buffer):
                row = [float(n) for n in row]
                img.append(row)

                if (i + 1) % self.ws == 0:
                    self.image_list.append(np.array(img))
                    # print(len(img))
                    img.clear()

        with open(label_file, 'r') as label_f:
            self.label_list = list(label_f.read())
            # print(len(self.label_list))

    def shuffle_lists(self, l1, l2):
        random.seed(4)
        mapIndexPosition = list(zip(l1, l2))
        random.shuffle(mapIndexPosition)
        l1, l2 = zip(*mapIndexPosition)
        return list(l1), list(l2)

    def __getitem__(self, index):
        # plt.imshow(self.image_list[index], 'gray')
        # plt.savefig('image.png')
        # plt.show()
        # print(self.image_list)
        # print(len(self.image_list))
        # print(self.label_list)
        # print(len(self.label_list))
        return torch.tensor(self.image_list[index]), torch.tensor(int(self.label_list[index]))

    def __len__(self):
        return len(self.label_list)

import torch.nn as nn


class dynamic_model(nn.Module):
    def __init__(self, H_in, W_in, num_kernels):
        super(dynamic_model, self).__init__()

        #ConV1
        C_in_1, C_out_1 = 1, num_kernels
        kernel_size_1 = 6
        self.layer1 = nn.Sequential(
            nn.Conv2d(int(C_in_1), int(C_out_1), kernel_size=kernel_size_1, stride=1, padding=0),
            nn.ReLU())
        H_out_1, W_out_1 = self.output_shape((H_in, W_in), kernel_size=kernel_size_1)

        #Conv2
        C_in_2, C_out_2 = C_out_1, num_kernels
        kernel_size_2 = 6
        self.layer2 = nn.Sequential(
            nn.Conv2d(int(C_in_2), int(C_out_2), kernel_size=kernel_size_2, stride=1, padding=0),
            nn.ReLU())
        H_out_2, W_out_2 = self.output_shape((H_out_1, W_out_1), kernel_size=kernel_size_2)

        # for Maxpooling layer1
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        H_out_Mx1, W_out_Mx1 = self.output_shape((H_out_2, W_out_2), kernel_size=3, stride=1)

        #Conv3
        C_in_3, C_out_3 = C_out_2, num_kernels
        kernel_size_3 = 6
        self.layer3 = nn.Sequential(
            nn.Conv2d(int(C_in_3), int(C_out_3), kernel_size=kernel_size_3, stride=1, padding=0),
            nn.ReLU())
        H_out_3, W_out_3 = self.output_shape((H_out_Mx1, W_out_Mx1), kernel_size=kernel_size_3)

        # for Maxpooling layer2
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        H_out_Mx2, W_out_Mx2 = self.output_shape((H_out_3, W_out_3), kernel_size=3, stride=1)

        self.fc1 = nn.Sequential(nn.Linear(int(C_out_3) * int(H_out_Mx2) * int(W_out_Mx2), 2))

    def forward(self, x):
        x = x.float()
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.maxpool1(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.maxpool2(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        return x

    def output_shape(self, input_size, kernel_size, stride=1, padding=0):
        H_in, W_in = input_size
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        return H_out, W_out

def config():
    print('\n\tInitializing experiment configurations...')
    exp_settings = {}
    exp_settings['batch_size'] = 128
    exp_settings['num_epochs'] = 100
    exp_settings['patience'] = 10
    exp_settings['window_size'] = 36
    exp_settings['space'] = {'lr': hp.uniform('lr', 0.00001, 0.001), 'num_kernels': 1 + hp.randint('num_kernels', 6)}
    exp_settings['folder'] = 'experiment_output918'
    if not os.path.exists(exp_settings['folder']): os.mkdir(exp_settings['folder'])
    exp_settings['datafile'] = exp_settings['folder'] + '/data_summary.txt'
    exp_settings['run_file'] = exp_settings['folder'] + '/eval_summary.txt'
    exp_settings['run_counter'] = 0
    exp_settings['glob_loss'] = 10
    return exp_settings


def load_all_data(exp_settings):
    def load_data(mode):
        dataset = ProtPep_dataset(ws=exp_settings['window_size'], mode=mode)
        return DataLoader(dataset, shuffle=True, batch_size=exp_settings['batch_size'])

    def save_details():
        for mode, loader in sets.items():
            print('Summarizing ', mode, ' set...')
            with open(exp_settings['datafile'], 'a') as f:
                f.write('\n' + mode + ' set contains ' + str(len(loader.dataset)) + ' samples')


    print('\n\tLoading datasets...')
    sets = {}
    sets['training'] = load_data('train')
    sets['validation'] = load_data('val')
    sets['test'] = load_data('test')
    save_details()
    return sets



ie = 0

def train(model, lr):
    global exp_settings, data, device

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1])).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_loss = 10
    patience_counter = exp_settings['patience']
    best_val = {}
    best_val['epoch_counter'] = 0
    best_val['val_loss_list'] = []
    best_val['train_loss_list'] = []
    for epoch in range(exp_settings['num_epochs']):
        epoch_train_loss, epoch_val_loss = [], []

        model.train()
        for (images, labels) in data['training']:
            inputs = Variable(images).to(device)
            inputs = inputs.unsqueeze(1)
            labels = Variable(labels.long()).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(loss)
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        best_val['train_loss_list'].append(sum(epoch_train_loss) / len(epoch_train_loss))

        val_outputs, val_labels, val_prob = [], [], []  # for metrics calculations
        model.eval()
        for (images, labels) in data['validation']:
            inputs = Variable(images).to(device)
            inputs = inputs.unsqueeze(1)
            labels = Variable(labels.long()).to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            epoch_val_loss.append(val_loss.item())
            out_max = outputs.detach()
            val_prob.append(out_max)
            out_max = torch.argmax(out_max, dim=1)
            val_outputs.append(out_max)
            val_labels.append(labels)
            global ie
            ie = ie + 1

        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        best_val['val_loss_list'].append(val_loss)

        if val_loss < max_loss:
            max_loss = val_loss
            patience_counter = 0
            best_val['val_outputs'] = torch.cat(val_outputs).cpu().numpy()
            best_val['val_labels'] = torch.cat(val_labels).cpu().numpy()
            best_val['output_prob'] = torch.cat(val_prob).cpu().numpy()[:, 1:]
            best_val['model_state'] = model.state_dict()
            best_val['epoch_counter'] = epoch
        if epoch == exp_settings['num_epochs'] - 1:
            max_loss = val_loss
            patience_counter = 0
            best_val['val_outputs'] = torch.cat(val_outputs).cpu().numpy()
            best_val['val_labels'] = torch.cat(val_labels).cpu().numpy()
            best_val['output_prob'] = torch.cat(val_prob).cpu().numpy()[:, 1:]
            best_val['model_state'] = model.state_dict()
            best_val['epoch_counter'] = epoch
        else:
            patience_counter += 1
        if patience_counter == exp_settings['patience']: break

    with open(exp_settings['folder'] + '/run' + str(exp_settings['run_counter']) + '.pickle', 'wb') as f:
        pickle.dump(best_val, f)
        # pickle.dump(model, f)
        torch.save(best_val['model_state'], 'save_modelsss/' + str(++ie) + 'net.pth')
    with open('smotemodel' + '/run' + str(exp_settings['run_counter']) + '.pickle', 'wb') as f1:
        #pickle.dump(best_val, f1)
        pickle.dump(model, f1)
        #torch.save(model, 'save_models/'+str(epoch)+'net.pth')
    #print(type(best_val['val_labels']))
    np.savetxt("result/"+str(exp_settings['run_counter'])+".csv", best_val['val_outputs'], delimiter=',')
    np.savetxt( str(exp_settings['run_counter']) + ".csv", best_val['val_labels'], delimiter=',')

    scores = {}
    # print(iter_dict['val_labels'])
    # print(iter_dict['val_outputs'])
    temp = confusion_matrix(best_val['val_labels'], best_val['val_outputs'])
    TN, FP, FN, TP = temp.ravel()
    # scores['misclassification_rate'] = (FP + FN) / (TN + FP + FN + TP ) # or 1-accuracy
    scores['sensitivity'] = TP / (FN + TP)  # aka sensitivity or recall
    # scores['false_pos_rate'] = FP / (TN + FP)
    scores['specificity'] = TN / (TN + FP)  # aka specificity
    precision = TP / (TP + FP)
    # scores['prevalence'] = (TP + FN) / (TN + FP + FN + TP )
    scores['f_score'] = (2 * scores['sensitivity'] * precision) / (scores['sensitivity'] + precision)

    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator == 0: mcc_denominator = 1
    scores['mcc'] = mcc_numerator / math.sqrt(mcc_denominator)
    scores['auc'] = roc_auc_score(best_val['val_labels'], best_val['output_prob'])
    scores['accuracy'] = (TP + TN) / (TN + FP + FN + TP)
    scores['conf_matrix'] = temp
    return best_val, scores


def pad(x):
    max_len = 11
    x = str(x)
    missing_len = max_len - len(x)
    x = x + (' ' * missing_len)
    return x


def add_header(file, score_dict):
    with open(file, 'a+') as f:
        f.write('\n|' + ('=' * (18 * 11)) + '|')
        f.write(
            '\n|' + pad('Iter') + '|' + pad('Status') + '|' + pad('Loss') + '|' + pad('So far') + '|' + pad('Runtime') +
            '|' + pad('LR') + '|' + pad('Nodes') + '|' + pad('Epochs') + '|')
        for key in score_dict.keys():
            if key == 'conf_matrix':
                f.write(pad('TN FP FN TP') + (' ' * 19) + '|')
            else:
                f.write(pad(key) + '|')
        f.write('\n|' + ('=' * (18 * 11)) + '|')


def save_to_file(file, count, status, loss, sofar, runtime, epochs, lr, num_kernels, score_dict):
    with open(file, 'a+') as f:
        f.write(('\n|' + pad(count) + '|' + pad(status) + '|' + pad(round(loss, 6)) + '|' + pad(round(sofar, 6)) + '|' +
                 pad(runtime) + '|' + pad(round(lr, 6)) + '|' + pad(num_kernels) + '|' + pad(epochs + 1) + '|'))
        for key, val in score_dict.items():
            if key == 'conf_matrix':
                f.write(str(val.ravel()))
            else:
                f.write(pad(round(val, 6)) + '|')
        f.write('\n|' + ('-' * (18 * 11)) + '|')


def obj_fn(space):
    global device, exp_settings
    start_time = time.time()
    lr = space['lr']
    num_kernels =int(math.pow(2, space['num_kernels']))
    model = dynamic_model(exp_settings['window_size'],32, num_kernels).to(device)
    iter_dict, iter_scores = train(model, lr)
    # torch.save(model, 'save_models/0net.pth')
    # print(iter_dict)
    # print(iter_dict['val_labels'])
    # print(iter_dict['val_outputs'])

    # iter_scores = calc_metrics(iter_dict)

    loss = -(iter_scores['auc'])

    if iter_scores['mcc'] <= -1.0:
        hp_status = STATUS_FAIL
        status = 'FAIL'
    else:
        hp_status = STATUS_OK
        if loss < exp_settings['glob_loss']:
            status = 'BEST'
            exp_settings['glob_loss'] = loss
        else:
            status = 'ACCEPT'

    runtime = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    if exp_settings['run_counter'] % 20 == 0: add_header(exp_settings['run_file'], iter_scores)
    save_to_file(exp_settings['run_file'], exp_settings['run_counter'], status, loss, exp_settings['glob_loss'],
                 runtime, iter_dict['epoch_counter'], lr, num_kernels, iter_scores)

    exp_settings['run_counter'] += 1

    return {'loss': loss, 'status': hp_status}


def run_trials():
    global exp_settings, data
    step = 1
    max_trials = 3  # have about 3 or 5 iterations to start with
    file_name = exp_settings['folder'] + '/trials_file.hyperopt'

    try:
        trials = pickle.load(open(file_name, 'rb'))
        print("\n\tFound saved Trials! Loading...")
        max_trials = len(trials.trials) + step
        print('Rerunning from {} to {} trials...'.format(len(trials.trials), len(trials.trials) + step))
    except:
        print("\n\tCreating new Trials!...")
        trials = Trials()

    best = fmin(fn=obj_fn,
                space=exp_settings['space'],
                algo=tpe.suggest,
                trials=trials,
                max_evals=max_trials)

    print('Best : ', best)

    with open(file_name, 'wb') as f:
        pickle.dump(trials, f)


def main():
    global exp_settings, device, data
    print('\nBayesian optimization of CNN hyperparameters, model for predicting PPIs .' +
          '\n-------------------------------------------------------------------------------------------------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    exp_settings = config()
    data = load_all_data(exp_settings)

    while True:
        run_trials()



if __name__ == '__main__':
    main()
