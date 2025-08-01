import torch
import torchvision
import os
import utils

#from imagenet_dataset import get_train_dataprovider, get_val_dataprovider
import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from fitness_function import fitnessFunction
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary
assert torch.cuda.is_available()

train_dataprovider, val_dataprovider = None, None
test_dataprovider = None
class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, cand, args):
    global train_dataprovider, val_dataprovider, test_provider
    if train_dataprovider is None:
        '''
        use_gpu = False
        train_transform, valid_transform = utils.data_transforms(args)
        trainset = torchvision.datasets.CIFAR10(root=os.path.join('./dataset/', 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root=os.path.join('./dataset/', 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=256,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        #train_dataprovider = get_train_dataprovider(
        #    args.train_batch_size, use_gpu=False, num_workers=0)
        #val_dataprovider = get_val_dataprovider(
        #    args.test_batch_size, use_gpu=False, num_workers=0)
        train_dataprovider = DataIterator(train_loader)
        val_dataprovider = DataIterator(val_loader)
        '''
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        generator1 = torch.Generator().manual_seed(42)
        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        trainset, valset = torch.utils.data.random_split(trainset, [42500, 7500], generator = generator1)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=100, shuffle=True, num_workers=2)

        testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        args.max_test_iters = len(val_loader)

        train_dataprovider = DataIterator(train_loader)
        val_dataprovider = DataIterator(val_loader)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters
    
    
    cand = list(cand)
    for num in range(len(cand)):
        if cand[num] == 12:
            cand.pop(num)
            cand.append(12)
    cand = tuple(cand)
    choice = [x//12 for x in cand]
    kchoice = [(x%12)//4 for x in cand]
    kernel_choice = [(x%12)%4 for x in cand]
    print('clear bn statics....')

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
    
    print('train bn with training set (BN sanitize) ....')
    model.train()
    
    for step in tqdm.tqdm(range(max_train_iters)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        output, _ = model(data, choice, kchoice, kernel_choice)
        del data, target, output
    
    top1 = 0
    top5 = 0
    total = 0
    print('starting test....')
    model.eval()
    for step in tqdm.tqdm(range(max_test_iters)):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        logits, _ = model(data, choice, kchoice, kernel_choice)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize

        del data, target, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total
    features, subnet = fitnessFunction.input_for_predictive_model(model, choice, kchoice, kernel_choice)

    if args.use_predictive_model:
        latency, power = fitnessFunction.predict_latency_power(features)
        power = power/1000
        print(f"[Predicted] Latency: {latency} ms, Power: {power} W")
    else:
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        latency, power = fitnessFunction.measure_latency_power_on_jetson(subnet, dummy_input)
        print(f"[Measured] Latency: {latency} ms, Power: {power} W")

    print('top1: {:.2f} top5: {:.2f}'.format(top1, top5))
    #Make sure the units are as required power in W latency in ms
    edp_min = 0.38452329
    edp_max = 2148262.951
    edp = latency*latency*power
    edp_norm = 100 * (edp - edp_min)/(edp_max - edp_min)
    # print("edp norm: ", edp_norm)
    fitness_top1 = 0.8 * top1 - 0.2 * edp_norm
    fitness_top5 = 0.8 * top5 - 0.2 * edp_norm

    print(f"Fitness Score: {fitness_top1:.4f}, {fitness_top5}")
    return fitness_top1, fitness_top5 
    # return top1, top5
    # add_model(choice, kchoice, cand)
    # try:
    #     latency,energy = run_mnsim(str(cand))
    #     latency = latency * 1e-6
    #     energy = energy * 1e-6
    #     latency_norm = 100*(latency-0.164)/(1.414-0.164)
    #     energy_norm = 100*(energy-0.033)/(63-0.033)
    #     print('latency: ' + str(latency) + ' Energy: ' + str(energy))
    #     #return top1**2/(latency*energy), top5**2/(latency*energy)
    #     return 0.8*top1-0.2*latency_norm*energy_norm, 0.8*top5-0.2*latency_norm*energy_norm
    # except:
    #     return False


def main():
    pass
