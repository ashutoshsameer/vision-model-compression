import argparse
from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
    parser.add_argument('--quant_method', default='linear', help='linear|minmax|log|tanh')
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=8, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
    parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

    parser.add_argument('--input_size', type=int, default=224, help='input size of image')
    parser.add_argument('--n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
    parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
    parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
    parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
    parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
    args = parser.parse_args()

    args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
    args.ngpu = len(args.gpu)
    misc.ensure_dir(args.logdir)
    args.model_root = misc.expand_user(args.model_root)
    args.data_root = misc.expand_user(args.data_root)
    args.input_size = 299 if 'inception' in args.type else args.input_size
    assert args.quant_method in ['linear', 'minmax', 'log', 'tanh']
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    assert torch.cuda.is_available(), 'no cuda'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)

    # TODO: add train model code here
    train_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=True, val=False, input_size=args.input_size)
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    model_raw = train_model(model_raw, train_ds, val_ds, epochs=20)

    args.ngpu = args.ngpu if is_imagenet else 1

    # quantize parameters
    if args.param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if args.bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits

            if args.quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=args.overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif args.quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif args.quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant)

    # quantize forward activation
    if args.fwd_bits < 32:
        model_raw = quant.duplicate_model_with_quant(model_raw, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                     counter=args.n_sample, type=args.quant_method)
        print(model_raw)
        val_ds_tmp = ds_fetcher(10, data_root=args.data_root, train=False, input_size=args.input_size)
        misc.eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=args.n_sample, is_imagenet=is_imagenet)

    # eval model
    acc1, acc5 = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

    torch.save(model_raw.state_dict(), f'./resnet18-{args.param_bits}-{args.fwd_bits}-{args.bn_bits}.pth' )
    # print sf
    print(model_raw)
    res_str = "type={}, quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        args.type, args.quant_method, args.param_bits, args.bn_bits, args.fwd_bits, args.overflow_rate, acc1, acc5)
    print(res_str)
    with open('acc1_acc5.txt', 'a') as f:
        f.write(res_str + '\n')


# def train_model(model, train_ds, epochs):
#     import tqdm
#     n_sample = 20
#     print(train_ds)
#     for idx, (data, target) in enumerate(tqdm.tqdm(train_ds, total=n_sample)):
# #         print(data.shape) # torch.Size([100, 3, 32, 32])
# #         print(target.shape) # torch.Size([100])
        
# #     print(model)
# #     print(train_ds)
# #     print(epochs)
#     return model

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader, epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 70, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
#             if i%10==0 and args.verbose:
#                 print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch, epochs, acc))
#         if best_acc<acc:
#             torch.save( model, 'resnet18-round%d.pth'%(args.round) )
#             best_acc=acc
        scheduler.step()
#     print("Best Acc=%.4f"%(best_acc))
    return model


if __name__ == '__main__':
    main()
