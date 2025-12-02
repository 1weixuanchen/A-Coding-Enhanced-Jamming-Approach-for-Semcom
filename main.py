import torch
import torchvision.transforms as transforms
from scm_main_net import SuperpositionNet
from train import train
from utils import init_seeds, data_loader, data_loader_ffhq
import os
import argparse
from eval import EVAL_proposed
import cka

def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    if not os.path.exists(config.rec_path):
        os.makedirs(config.rec_path)
    if not os.path.exists(config.data_path):
        os.makedirs(config.data_path)

def main(config):
    # set seeds
    init_seeds()

    # prepare data
    transform_train = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(32, scale=(0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    transform_test = [transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                      ]

    train_loader, test_loader = data_loader(config.dataset_path, config.batch_size, transform_train, transform_test)

    # FFHQ / MNIST

    mnist_transform_train = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(64, scale=(0.8, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    mnist_transform_test = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]

    train_loader2, test_loader2 = data_loader_ffhq(config.dataset_path_mnist, config.batch_size,
                                                    mnist_transform_train,
                                                    mnist_transform_test)

    if config.mode == 'train':
        start_train(config, train_loader, test_loader, train_loader2, test_loader2)
    elif config.mode == 'test':
        start_test(config, test_loader, test_loader2)

def start_train(config, train_loader, test_loader, train_loader2, test_loader2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.net == 'scm':
        print('Superposition is {}.'.format(config.sp_mode))
        print('Channel use is: {}.'.format(config.channel_use))
        print('The SNR of the eve user: {}.'.format(config.snr_train_eve))
        print('a = {}.'.format(config.a))

        """"""
        config.net = 'scm'
        sp_model = SuperpositionNet(config, device).to(device)

        """Phase 1"""

        for param in sp_model.parameters():
            param.requires_grad = True

        for param in sp_model.eve_decoder.parameters():
            param.requires_grad = False

        for param in sp_model.leg_inner_decoder.parameters():
            param.requires_grad = False

        config.phase = '1'
        config.train_iters = 100
        config.lr = 2e-4
        print('Phase 1 training')
        train(config, sp_model, train_loader, test_loader, device, train_loader2, test_loader2, config.phase)

        """Phase 2"""
        sp_model = SuperpositionNet(config, device).to(device)

        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}_alpha1_{}_alpha2_{}_alpha3_{}_4_13_ffhq.pth.tar'.format(
            config.net,
            config.snr_train_eve,
            config.snr_train_leg,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a,
            config.alpha_1,
            config.alpha_2,
            config.alpha_3
        )

        sp_model.load_state_dict(torch.load(config.model_path + model_name))

        print('LOAD Phase 1 Model')

        for param in sp_model.parameters(): #
            param.requires_grad = False

        for param in sp_model.eve_decoder.parameters(): # 第2阶段只训练窃听者作为一个模拟的对手
            param.requires_grad = True

        config.phase = '2'
        config.train_iters = 100
        config.lr = 2e-4
        print('Phase 2 training.')
        train(config, sp_model, train_loader, test_loader, device, train_loader2, test_loader2,config.phase)

        """Phase 3"""

        model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}_alpha1_{}_alpha2_{}_alpha3_{}_4_13_ffhq.pth.tar'.format(
            config.net,
            config.snr_train_eve,
            config.snr_train_leg,
            config.channel_use,
            config.sp_mode,
            config.phase,
            config.a,
            config.alpha_1,
            config.alpha_2,
            config.alpha_3
        )

        print(model_name,"load_model_name")

        sp_model = SuperpositionNet(config, device).to(device)
        sp_model.load_state_dict(torch.load(config.model_path + model_name))

        print('LOAD Phase 2 Model')

        for param in sp_model.parameters():
            param.requires_grad = False

        for param in sp_model.basic_encoder.parameters():
            param.requires_grad = True
        #
        for param in sp_model.leg_decoder.parameters():
            param.requires_grad = True
        #
        for param in sp_model.leg_inner_decoder.parameters():
            param.requires_grad = True
        #
        for param in sp_model.enhancement_encoder.parameters():
            param.requires_grad = True

        config.phase = '3'

        print('Phase 3 training.')
        config.train_iters = 200
        config.lr = 5e-5
        train(config, sp_model, train_loader, test_loader, device, train_loader2, test_loader2,config.phase)


def start_test(config, test_loader, test_loader2):
    """Testing phase-3 models"""
    config.phase = '3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SuperpositionNet(config, device).to(device)

    model_name = '/{}_SNR{:.1f}_{:.1f}_Trans{:d}_{}_phase{}_{}_alpha1_{}_alpha2_{}_alpha3_{}_ffhq.pth.tar'.format(
        config.net,
        config.snr_train_eve,
        config.snr_train_leg,
        config.channel_use,
        config.sp_mode,
        config.phase,
        config.a,
        config.alpha_1,
        config.alpha_2,
        config.alpha_3
    )

    print(model_name)
    net.load_state_dict(torch.load(config.model_path + model_name))
    acc_total_leg, acc_total_eve, psnr_total_leg, psnr_total_eve, _, _ = EVAL_proposed(net, test_loader, test_loader2,
                                                                                       device, config, -1)

    print('Leg receiver: acc: {:.4f}, psnr: {:.4f}; eve receiver: acc: {:.4f}, psnr: {:.4f}'.format(
        acc_total_leg, psnr_total_leg, acc_total_eve, psnr_total_eve
    ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--channel_use', type=int, default=int(128))
    parser.add_argument('--net', type=str,
                        default='scm')
    parser.add_argument('--sp_mode', type=str, default='4and4')
    parser.add_argument('--a', type=float, default=0.49)

    parser.add_argument('--phase', type=str, default='3')
    parser.add_argument('--tradeoff', type=float, default=1e-2)
    parser.add_argument('--alpha_1', type=float, default=1.0)
    parser.add_argument('--alpha_2', type=float, default=1.5)
    parser.add_argument('--alpha_3', type=float, default=0.01)

    parser.add_argument('--snr_train_eve', type=float, default=10)
    parser.add_argument('--snr_train_leg', type=float, default=10)

    parser.add_argument('--snr_test_eve', type=float, default=10)
    parser.add_argument('--snr_test_leg', type=float, default=10)

    parser.add_argument('--train_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='train')

    # misc
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--rec_path', type=str, default='./rec')
    parser.add_argument('--dataset_path', type=str, default='/home/ubuntu/xxx')
    parser.add_argument('--dataset_path_mnist', type=str, default='/home/ubuntu/xxx')
    parser.add_argument('--data_path', type=str, default='./trainingPlot')

    config = parser.parse_args()

    mischandler(config)
    main(config)
