import torch
from torch import optim, nn
import pandas as pd
from tqdm import tqdm
import cka
from eval import EVAL_proposed

class CKALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature1, feature2):

        feature1 = feature1.view(feature1.size(0), -1).detach().cpu().numpy()
        feature2 = feature2.view(feature2.size(0), -1).detach().cpu().numpy()

        gram_x = cka.gram_linear(feature1)
        gram_y = cka.gram_linear(feature2)

        loss = cka.cka(gram_x, gram_y)

        return torch.tensor(loss, requires_grad=True)

def train(config, net, train_iter, test_iter, device , train_iter2 , test_iter2 , training_phase):

    loss_mse = nn.MSELoss()
    loss_MI = CKALoss()

    results_leg = {'epoch': [], 'loss': [],  'psnr': [], 'mse': []}
    results_eve = {'epoch': [], 'loss': [],  'psnr': [], 'mse': []}
    resi_track = {'epoch': [], 'residual': []}

    epochs = config.train_iters

    ignored_params = (
            list(map(id, net.basic_encoder.prob_convs.parameters())) +
            list(map(id, net.enhancement_encoder.prob_convs.parameters())) +
            list(map(id, net.leg_inner_decoder.prob_convs.parameters())) +
            list(map(id, net.leg_decoder.parameters())
                 )
    )

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, base_params)},
        {'params': filter(lambda p: p.requires_grad, net.basic_encoder.prob_convs.parameters()), 'lr': config.lr * 60},
        {'params': filter(lambda p: p.requires_grad, net.enhancement_encoder.prob_convs.parameters()),
         'lr': config.lr * 60},
        {'params': filter(lambda p: p.requires_grad, net.leg_inner_decoder.prob_convs.parameters()),
         'lr': config.lr * 60},
        {'params': filter(lambda p: p.requires_grad, net.leg_decoder.parameters()), 'lr': config.lr * 1}
    ], lr=config.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs+1, T_mult=1, eta_min=1e-5,
                                                                     last_epoch=-1)

    for epoch in range(epochs):

        epoch_loss = []
        net.train()

        for i, ((X1, _), (X2)) in enumerate(tqdm(zip(train_iter, train_iter2))):

            X1 = X1.to(device)
            X2 = X2.to(device)

            optimizer.zero_grad()

            u_resi, z, z_hat_leg, z_hat_eve, rec_leg, rec_eve , z_v , z_v_leg , z_u = net(X1, X2)

            mse_loss_eve = loss_mse(rec_eve, X1)
            mse_loss_leg = loss_mse(rec_leg, X1)
            mse_loss_inner_code = loss_mse(z_v, z_v_leg)
            MI_loss_zu_zv = loss_MI(z_v, z_u)

            if config.phase == '1':

                loss = mse_loss_leg + MI_loss_zu_zv * config.alpha_3

            elif config.phase == '2':

                loss = mse_loss_eve

            elif config.phase == '3':

                loss = config.alpha_1 * mse_loss_inner_code + mse_loss_leg - config.alpha_2 * mse_loss_eve + MI_loss_zu_zv * config.alpha_3

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.cpu().item())

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)

        if config.phase == '3':

            if (epoch + 1) % 2 == 0 :

                acc_leg, acc_eve, psnr_leg, psnr_eve, mse_leg, mse_eve = EVAL_proposed(net, test_iter, test_iter2, device,
                                                                                       config, epoch)

                print('[epoch: {:d}/loss: {:.6f}] \n LEG receiver: acc: {:.3f}, psnr: {:.3f}, mse:{:.5f}'
                      .format(epoch, loss, acc_leg, psnr_leg, mse_leg))
                print('EVE receiver: acc: {:.3f}, psnr: {:.3f}, mse: {:.5f}'
                      .format(acc_eve, psnr_eve, mse_eve))

                results_leg['epoch'].append(epoch)
                results_leg['loss'].append(loss)
                results_leg['psnr'].append(psnr_leg)
                results_leg['mse'].append(mse_leg)

                results_eve['epoch'].append(epoch)
                results_eve['loss'].append(loss)
                results_eve['psnr'].append(psnr_eve)
                results_eve['mse'].append(mse_eve)

                resi_track['epoch'].append(epoch)
                resi_track['residual'].append(0)

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

                torch.save(net.state_dict(), config.model_path + model_name)
                print('Successfully save present model!')

                print(f"Model saved at epoch {epoch + 1} with psnr_leg: {psnr_leg:.3f}, psnr_eve: {psnr_eve:.3f}")

        else :

            if (epoch + 1) % 5 == 0 or epoch == 0 :

                acc_leg, acc_eve, psnr_leg, psnr_eve, mse_leg, mse_eve = EVAL_proposed(net, test_iter, test_iter2,
                                                                                       device,
                                                                                       config, epoch)

                print('[epoch: {:d}/loss: {:.6f}] \n LEG receiver: acc: {:.3f}, psnr: {:.3f}, mse:{:.5f}'
                      .format(epoch, loss, acc_leg, psnr_leg, mse_leg))
                print('EVE receiver: acc: {:.3f}, psnr: {:.3f}, mse: {:.5f}'
                      .format(acc_eve, psnr_eve, mse_eve))

                results_leg['epoch'].append(epoch)
                results_leg['loss'].append(loss)
                results_leg['psnr'].append(psnr_leg)
                results_leg['mse'].append(mse_leg)

                results_eve['epoch'].append(epoch)
                results_eve['loss'].append(loss)
                results_eve['psnr'].append(psnr_eve)
                results_eve['mse'].append(mse_eve)

                resi_track['epoch'].append(epoch)
                resi_track['residual'].append(0)

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

                torch.save(net.state_dict(), config.model_path + model_name)
                print('Successfully save present model!')



