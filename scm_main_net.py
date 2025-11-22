from torch import nn
import torch
from torch.nn import init
from basic_module import awgn, DepthToSpace, Decoder_Recon, normalize, Encoder , Encoder_Inner
from torch.nn.functional import gumbel_softmax
import math

def modulation(probs, order, device, p=1):
    order_sqrt = int(order ** 0.5)
    prob_z = gumbel_softmax(probs, hard=False)
    discrete_code = gumbel_softmax(probs, hard=True, tau=1.7)

    if order_sqrt == 2:
        const = [1, -1]
    elif order_sqrt == 4:
        const = [-3, -1, 1, 3]
    elif order_sqrt == 8:
        const = [-7, -5, -3, -1, 1, 3, 5, 7]
    elif order_sqrt == 16:
        const = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]

    const = torch.tensor(const, dtype=torch.float).to(device)

    ave_p = torch.mean(const ** 2)
    const = const / ave_p ** 0.5

    temp = discrete_code * const
    output = torch.sum(temp, dim=2)

    return output, prob_z


class BasicEncoder(nn.Module):
    def __init__(self, config, device, order):
        super(BasicEncoder, self).__init__()

        self.device = device
        self.config = config
        self.order = order

        if self.order == 4:
            self.num_category = 2
        elif self.order == 16:
            self.num_category = 4

        self.encoder = Encoder(config)
        self.ave_pool = nn.AvgPool2d(2)
        self.prob_convs = nn.Sequential(
            nn.Linear(config.channel_use * 2, config.channel_use * 2 * self.num_category),
            nn.PReLU()
        )

    def reparameterize(self, probs):
        code, probs = modulation(probs, self.order, self.device)
        return code, probs

    def forward(self, x):
        u = self.encoder(x)
        u = self.ave_pool(u).reshape(x.shape[0], -1)
        z_u = self.prob_convs(u).reshape(x.shape[0], -1, self.num_category)
        z_u, probs = self.reparameterize(z_u)
        return u, z_u, probs


class EnhancementEncoder(nn.Module):
    def __init__(self, config, device, order):
        super(EnhancementEncoder, self).__init__()

        self.config = config
        self.device = device
        self.order = order

        if self.order == 4:
            self.num_category = 2
        elif self.order == 16:
            self.num_category = 4
        elif self.order == 64:
            self.num_category = 8
        elif self.order == 256:
            self.num_category = 16

        self.encoder = Encoder_Inner(config)
        self.ave_pool = nn.AvgPool2d(2)
        self.prob_convs = nn.Sequential(
            nn.Linear(config.channel_use * 2, config.channel_use * 2 * self.num_category),
            nn.PReLU()
        )

    def reparameterize(self, probs):

        code, probs = modulation(probs, self.order, self.device)
        return code, probs

    def forward(self, x ):

        v = self.encoder(x)
        v = self.ave_pool(v).reshape(x.shape[0], -1)
        u_residual = v
        z_v = self.prob_convs(v).reshape(x.shape[0], -1, self.num_category)
        z_v, probs = self.reparameterize(z_v)

        return u_residual, z_v, probs


class LegDecoder(nn.Module):

    def __init__(self, config):
        super(LegDecoder, self).__init__()
        self.config = config

        self.rec_decoder = Decoder_Recon(config)

        self.layer_width = int(config.channel_use * 2 / 8)
        self.Half_width = int(config.channel_use * 2 / 2)

    def forward(self, z_hat):
        rec = self.rec_decoder(z_hat.reshape(z_hat.shape[0], -1, 4, 4))
        return rec

class EveDecoder(nn.Module):
    def __init__(self, config):
        super(EveDecoder, self).__init__()

        self.rec_decoder = Decoder_Recon(config)

        self.layer_width = int(config.channel_use * 2 / 8)
        self.Half_width = int(config.channel_use * 2 / 2)

    def forward(self, z_hat):
        rec = self.rec_decoder(z_hat.reshape(z_hat.shape[0], -1, 4, 4))
        return rec


class SuperpositionNet(nn.Module):
    def __init__(self, config, device):
        super(SuperpositionNet, self).__init__()
        self.config = config
        self.device = device

        if self.config.sp_mode == '4and4':
            low_order = 4
            high_order = 4
            self.a = self.config.a

        elif self.config.sp_mode == '4and16':
            low_order = 4
            high_order = 16
            self.a = self.config.a

        elif self.config.sp_mode == '16and16':
            low_order = 16
            high_order = 16
            self.a = self.config.a

        elif self.config.sp_mode == '16and4':
            low_order = 16
            high_order = 4
            self.a = self.config.a

        elif self.config.sp_mode == '4and64':
            low_order = 4
            high_order = 64
            self.a = self.config.a

        elif self.config.sp_mode == '4and256':
            low_order = 4
            high_order = 256
            self.a = self.config.a

        self.basic_encoder = BasicEncoder(self.config, self.device, low_order)

        self.enhancement_encoder = EnhancementEncoder(self.config, self.device, high_order)

        self.leg_inner_decoder = EnhancementEncoder(self.config, self.device, high_order)

        self.leg_decoder = LegDecoder(self.config)

        self.eve_decoder = EveDecoder(self.config)

        self.high_order = high_order

        self.initialize_weights()

    def initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x , x_fake):

        u, z_u, probs_u = self.basic_encoder(x)

        u_residual, z_v, probs_v = self.enhancement_encoder(x_fake)

        z = (self.a ** 0.5) * z_u + ((1 - self.a) ** 0.5) * z_v

        _, z_transmitter_for_eve = normalize(z)

        power_emp_z = torch.mean(torch.abs(z) ** 2)

        if self.config.mode == 'train':

            z_hat_leg = awgn(self.config.snr_train_leg, z, self.device, p=power_emp_z.item())
            z_hat_eve = awgn(self.config.snr_train_eve, z_transmitter_for_eve, self.device)

        elif self.config.mode == 'test':

            z_hat_leg = awgn(self.config.snr_test_leg, z, self.device, p=power_emp_z.item())
            z_hat_eve = awgn(self.config.snr_test_eve, z_transmitter_for_eve, self.device)

        if self.config.phase in ['1', '2'] :

            z_v_leg = z_v

        elif self.config.phase in ['3'] :

            _, z_v_leg, _ = self.leg_inner_decoder(x_fake)

        z_u_leg = ( z_hat_leg - (((1 - self.a) ** 0.5) * z_v_leg) ) / (self.a ** 0.5)

        _, z_hat_leg_new = normalize(z_u_leg)

        rec_leg = self.leg_decoder(z_hat_leg_new)

        if self.training == False :

            rec_eve = self.leg_decoder(z_hat_eve)

        else :

            rec_eve = self.eve_decoder(z_hat_eve)

        return u_residual, z, z_hat_leg, z_hat_eve, rec_leg, rec_eve , z_v , z_v_leg , z_u

