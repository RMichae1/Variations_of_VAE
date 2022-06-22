"""
Created on Tue Feb 18 13:13:07 2020

@author: JAKS
"""
import torch
from torch import nn
from ..utils import smooth_one_hot
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.nn import functional as F
import itertools
import time


def generate_all_labels(x, y_dim):
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return generated.float()


def multilabel_generate_permutations(x, label_dim, multilabel):
    def generate_all_labels(x, y_dim):
        def batch(batch_size, label):
            labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
            y = torch.zeros((batch_size, y_dim))
            y.scatter_(1, labels, 1)
            return y.type(torch.LongTensor)
        generated = torch.cat([batch(1, i) for i in range(y_dim)])

        if x.is_cuda:
            generated = generated.cuda()

        return generated.float()
    batch_size = x.size(0)
    permuts = [list(p) for p in itertools.product(list(range(label_dim)), repeat=multilabel)]
    labels = torch.stack([generate_all_labels(x, label_dim)[per].view(1, -1).squeeze(0) for per in permuts])
    if x.is_cuda:
        return torch.repeat_interleave(labels, batch_size, dim=0).cuda().float()
    else:
        return torch.repeat_interleave(labels, batch_size, dim=0)


def sample_gaussian(mu, logsigma): #reparametrization trick
    std = torch.exp(logsigma)
    eps = torch.randn_like(std)
    return mu + eps * std


def KLdivergence(mu, sigma):
    return 0.5*torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)


def log_standard_categorical(p):
    with torch.no_grad():
        prior = F.softmax(torch.ones_like(p), dim=1)
        cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    return cross_entropy


def idx_for_each_label(labels, multilabel, num_label_cols):
    keep_idx = []
    cols_per_label = num_label_cols // multilabel
    for i in range(multilabel):
        if cols_per_label > 1:
            y_val = labels[:,i*cols_per_label:i*cols_per_label+cols_per_label]
        else:
            y_val =labels[:,i]
        keep_idx.append(torch.unique(torch.where(y_val>=0)[0]))
    return keep_idx


def cross_entropy_loss(pred, y):
    return -torch.sum(y * torch.log(pred+1e-11), dim=1).mean()


class Classifier(nn.Module):
    def __init__(self, layer_size):
        super(Classifier, self).__init__()
        [input_dim, h_dim, y_dim] = layer_size
        self.dense = nn.Linear(input_dim, h_dim)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.1)

        self.logits = nn.Linear(h_dim, y_dim)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.constant_(self.logits.bias, 0.1)

        self.network = nn.Sequential(self.dense,
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     self.logits,
                                     nn.Softmax(dim=1))

    def forward(self, x):
        x = self.network(x)
        return x

class Regressor(nn.Module):
    def __init__(self, layer_size):
        super(Regressor, self).__init__()
        [input_dim, h_dim, y_dim] = layer_size
        self.dense = nn.Linear(input_dim, h_dim)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.1)

        self.logits = nn.Linear(h_dim, y_dim)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.constant_(self.logits.bias, 0.1)

        self.network = nn.Sequential(self.dense,
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     self.logits)

    def forward(self, x):
        x = self.network(x)
        return x


class VAE_bayes(nn.Module):
    """
    We differentiate between SSVAE and SSCVAE/CVAE bc they have discretized labels
    """
    def __init__(self,
                 layer_sizes,
                 alphabet_size,
                 z_samples = 1,
                 dropout = 0.5,
                 use_bayesian = True,
                 num_patterns = 4,
                 cw_inner_dimension = 40,
                 use_param_loss = True,
                 use_sparse_interactions = False,
                 rws = False,
                 conditional_data_dim=False,
                 multilabel=0,
                 seq2yalpha=0,
                 z2yalpha=0,
                 label_pred_layer_sizes = None,
                 pred_from_latent=False,
                 pred_from_seq=False,
                 warm_up = 0,
                 batchnorm = False,
                 auxiliary_ELBO_loss = False,
                 device='cuda'):
        super(VAE_bayes, self).__init__()
        self.layer_sizes = layer_sizes
        self.alphabet_size = alphabet_size
        self.max_sequence_length = self.layer_sizes[0] // self.alphabet_size
        self.cw_inner_dimension  = cw_inner_dimension
        self.device = device
        self.dropout = dropout
        self.nb_patterns = num_patterns
        self.dec_h2_dim = layer_sizes[-2]
        self.z_samples = z_samples
        self.use_bayesian = use_bayesian
        self.use_param_loss = use_param_loss
        self.use_sparse_interactions = use_sparse_interactions
        self.rws = rws
        self.warm_up = warm_up
        self.warm_up_scale = 0
        self.conditional_data_dim = conditional_data_dim
        self.multilabel = multilabel
        self.label_pred_layer_sizes = label_pred_layer_sizes
        self.pred_from_latent = pred_from_latent
        self.pred_from_seq = pred_from_seq
        self.seq2yalpha = seq2yalpha
        self.z2yalpha = z2yalpha
        self.batchnorm = batchnorm
        self.auxiliary_ELBO_loss = auxiliary_ELBO_loss
        self.device = device
        self.latent_dim = layer_sizes.index(min(layer_sizes))
        # Encoder
        # encoder neural network prior to mu and sigma
        input_dim = self.max_sequence_length*alphabet_size
        self.enc_fc1 = nn.Linear(input_dim, layer_sizes[1])
        nn.init.xavier_normal_(self.enc_fc1.weight)
        nn.init.constant_(self.enc_fc1.bias, 0.1)
        self.enc_fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        nn.init.xavier_normal_(self.enc_fc2.weight)
        nn.init.constant_(self.enc_fc2.bias, 0.1)
        if batchnorm:
            self.encode_layers = nn.Sequential(self.enc_fc1,
                                               nn.BatchNorm1d(layer_sizes[1]),
                                               nn.ReLU(inplace = True),
                                               self.enc_fc2,
                                               nn.BatchNorm1d(layer_sizes[2]),
                                               nn.ReLU(inplace = True))
        else:
            self.encode_layers = nn.Sequential(self.enc_fc1,
                                               nn.ReLU(inplace = True),
                                               self.enc_fc2,
                                               nn.ReLU(inplace = True))
        # encode mu from h_dim to z_dim deterministically (reparameterization trick)
        self.encode_mu = nn.Linear(layer_sizes[2], layer_sizes[3])
        nn.init.xavier_normal_(self.encode_mu.weight)
        nn.init.constant_(self.encode_mu.bias, 0.1)
        # encode sigma from h_dim to z_dim deterministically (reparameterization trick)
        self.encode_logsigma = nn.Linear(layer_sizes[2], layer_sizes[3])
        nn.init.xavier_normal_(self.encode_logsigma.weight)
        nn.init.constant_(self.encode_logsigma.bias, -10)
        # Decoder
        # weights layer 1
        z_dim = layer_sizes[3]
        # Auxiliary loss approximation NN addition
        if self.auxiliary_ELBO_loss:
            self.auxiliary_nn = nn.Sequential(nn.Linear(z_dim, 12),
                                            nn.Dropout(p=0.2),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(12, 1))
        # if sparse interactions and other tricks are enabled initiate variables
        if self.use_sparse_interactions:
            # sparse interaction and dict from deepsequence paper
            self.mu_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
            nn.init.zeros_(self.mu_S)
            self.logsigma_S = nn.Parameter(torch.Tensor(int(layer_sizes[5] / self.nb_patterns), self.max_sequence_length))
            nn.init.constant_(self.logsigma_S, -10)
            self.mu_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
            nn.init.xavier_normal_(self.mu_C)
            self.logsigma_C = nn.Parameter(torch.Tensor(cw_inner_dimension, alphabet_size))
            nn.init.constant_(self.logsigma_C, -10)
            # inverse temperature term from deepsequence paper
            self.mu_l = nn.Parameter(torch.Tensor([1]))
            self.logsigma_l = nn.Parameter(torch.Tensor([-10.0]))
            # alter output shape for sparse interactions
            W3_output_shape = cw_inner_dimension * self.max_sequence_length
        if not self.use_sparse_interactions:
            W3_output_shape = alphabet_size * self.max_sequence_length
        # bayesian network
        if self.use_bayesian:
            self.mu_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
            nn.init.xavier_normal_(self.mu_W1_dec)
            self.logsigma_W1_dec = nn.Parameter(torch.Tensor(z_dim, layer_sizes[4]))
            nn.init.constant_(self.logsigma_W1_dec, -10)
            self.mu_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
            nn.init.constant_(self.mu_b1_dec, 0.1)
            self.logsigma_b1_dec = nn.Parameter(torch.Tensor(layer_sizes[4]))
            nn.init.constant_(self.logsigma_b1_dec, -10)
            # weights layer 2
            self.mu_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
            nn.init.xavier_normal_(self.mu_W2_dec)
            self.logsigma_W2_dec = nn.Parameter(torch.Tensor(layer_sizes[4], layer_sizes[5]))
            nn.init.constant_(self.logsigma_W2_dec, -10)
            self.mu_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
            nn.init.constant_(self.mu_b2_dec, 0.1)
            self.logsigma_b2_dec = nn.Parameter(torch.Tensor(layer_sizes[5]))
            nn.init.constant_(self.logsigma_b2_dec, -10)
            # weights layer 3
            self.mu_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
            nn.init.xavier_normal_(self.mu_W3_dec)
            self.logsigma_W3_dec = nn.Parameter(torch.Tensor(layer_sizes[5], W3_output_shape))
            nn.init.constant_(self.logsigma_W3_dec, -10)
            self.mu_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.mu_b3_dec, 0.1)
            self.logsigma_b3_dec = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.logsigma_b3_dec, -10)
        # non-bayesian network
        h1_decoder_layers = []
        h2_decoder_layers = []
        if not self.use_bayesian:
            # decoder
            self.h1_dec = nn.Linear(z_dim, layer_sizes[4])
            nn.init.xavier_normal_(self.h1_dec.weight)
            nn.init.constant_(self.h1_dec.bias, 0.1)
            h1_decoder_layers.append(self.h1_dec)
            if batchnorm:
                h1_decoder_layers.append(nn.BatchNorm1d(layer_sizes[4]))
            h1_decoder_layers.append(nn.ReLU())
            if not batchnorm:
                h1_decoder_layers.append(nn.Dropout(self.dropout))
            self.h2_dec = nn.Linear(layer_sizes[4], layer_sizes[5])
            nn.init.xavier_normal_(self.h2_dec.weight)
            nn.init.constant_(self.h2_dec.bias, 0.1)
            h2_decoder_layers.append(self.h2_dec)
            h2_decoder_layers.append(nn.Sigmoid())
            if batchnorm:
                h2_decoder_layers.append(nn.BatchNorm1d(layer_sizes[5]))
            if not batchnorm:
                h2_decoder_layers.append(nn.Dropout(self.dropout))
            self.h3_dec = nn.Linear(layer_sizes[5], W3_output_shape, bias = False)
            nn.init.xavier_normal_(self.h3_dec.weight)
            self.b3 = nn.Parameter(torch.Tensor(alphabet_size * self.max_sequence_length))
            nn.init.constant_(self.b3, -10)
        self.h1_decoder_network = nn.Sequential(*h1_decoder_layers)
        self.h2_decoder_network = nn.Sequential(*h2_decoder_layers)
        if pred_from_latent:
            self.z_label_pred = nn.ModuleList([Regressor([layer_sizes[3], label_pred_layer_sizes[0], 1]) for i in range(multilabel)])
        self.to(device)

    def predict_label(self, x, i=0):
        x = F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float).cuda() if self.device=='cuda' else F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float)
        assert torch.isfinite(x).all() == True
        return self.label_pred_model[i](x)

    def predict_label_from_z(self, x, i=0):
        return self.z_label_pred[i](x)

    def encoder(self, x, labels):
        x = F.one_hot(x.to(torch.int64), self.alphabet_size).flatten(1).to(torch.float)
        h = self.encode_layers(x)
        mu = self.encode_mu(h)
        logvar = self.encode_logsigma(h)
        return mu+1e-6, logvar+1e-6

    def decoder(self, z, labels):
        # if bayesian sample new weights
        if self.use_bayesian:
            # weights and bias for layer 1
            W1 = sample_gaussian(self.mu_W1_dec, self.logsigma_W1_dec)
            b1 = sample_gaussian(self.mu_b1_dec, self.logsigma_b1_dec)
            W2 = sample_gaussian(self.mu_W2_dec, self.logsigma_W2_dec)
            b2 = sample_gaussian(self.mu_b2_dec, self.logsigma_b2_dec)
            W3 = sample_gaussian(self.mu_W3_dec, self.logsigma_W3_dec)
            b3 = sample_gaussian(self.mu_b3_dec, self.logsigma_b3_dec)
        else:
            W3 = self.h3_dec.weight
            b3 = self.b3

        # if sparse interactions perform linear operations
        if self.use_sparse_interactions:
            l = sample_gaussian(self.mu_l, self.logsigma_l)
            S = sample_gaussian(self.mu_S, self.logsigma_S)
            C = sample_gaussian(self.mu_C, self.logsigma_C)
            S = torch.sigmoid(S.repeat(self.nb_patterns, 1))
            W3 = W3.view(self.dec_h2_dim * self.max_sequence_length, -1)
            W_out = W3 @ C
            W_out = W_out.view(-1, self.max_sequence_length, self.alphabet_size)
            W_out = W_out * S.unsqueeze(2)
            W_out = W_out.view(-1, self.max_sequence_length * self.alphabet_size)
        if not self.use_sparse_interactions and not self.use_bayesian:
            W_out = W3.t()
        if not self.use_sparse_interactions and self.use_bayesian:
            W_out = W3

        if self.use_bayesian:
            h1 = nn.functional.relu(nn.functional.linear(z, W1.t(), b1))
            h2 = torch.sigmoid(nn.functional.linear(h1, W2.t(), b2))
        else:
            h1 = self.h1_decoder_network(z)
            h2 = self.h2_decoder_network(h1)

        h3 = nn.functional.linear(h2, W_out.t(), b3)
        if self.use_sparse_interactions:
            h3 = h3 * torch.log(1 + l.exp())
        h3 = h3.view((-1, self.max_sequence_length, self.alphabet_size))
        px_z = torch.distributions.Categorical(logits=h3)
        h3 = nn.functional.log_softmax(h3, -1)
        return h1, h2, h3, px_z

    def recon_loss(self, recon_x, x):
        # How well do input x and output recon_x agree?
        recon_x = recon_x.view(self.z_samples, -1, recon_x.size(1), self.alphabet_size).permute(1, 2, 0, 3)
        x = x.unsqueeze(-1).expand(-1, -1, self.z_samples)
        smooth_target = smooth_one_hot(x, self.alphabet_size)
        loss = -(smooth_target * recon_x).sum(-1)
        loss = loss.mean(-1).sum(-1)
        return loss

    def kld_loss(self, encoded_distribution):
        prior = Normal(torch.zeros_like(encoded_distribution.mean), torch.ones_like(encoded_distribution.variance))
        kld = kl_divergence(encoded_distribution, prior).sum(dim = 1)
        return kld

    def auxiliary_loss(self, approx_ELBO, ELBO, weight=1.):
        mse_loss = nn.MSELoss()
        aux_loss = mse_loss(approx_ELBO, ELBO) * weight
        return aux_loss

    def protein_logp(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
        kld = self.kld_loss(encoded_distribution)
        z = encoded_distribution.rsample()
        recon_x = self.decoder(z, labels)[2].permute(0, 2, 1)
        logp = F.nll_loss(recon_x, x, reduction = "none").mul(-1).sum(1)
        elbo = logp + kld
        # amino acid probabilities are independent conditioned on z
        return elbo, logp, kld

    def global_parameter_kld(self):
        global_kld = 0
        global_kld += -KLdivergence(self.mu_W1_dec, self.logsigma_W1_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b1_dec, self.logsigma_b1_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_W2_dec, self.logsigma_W2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b2_dec, self.logsigma_b2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_W3_dec, self.logsigma_W3_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b3_dec, self.logsigma_b3_dec.mul(0.5).exp())
        if not self.use_sparse_interactions:
            return global_kld
        global_kld += -KLdivergence(self.mu_C, self.logsigma_C.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_S, self.logsigma_S.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_l, self.logsigma_l.mul(0.5).exp())
        return global_kld

    def global_parameter_kld1(self):
        global_kld = 0
        global_kld += torch.sum(self.kld_loss(Normal(self.mu_W1_dec, self.logsigma_W1_dec.mul(0.5).exp())))
        global_kld += -KLdivergence(self.mu_b1_dec, self.logsigma_b1_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_W2_dec, self.logsigma_W2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b2_dec, self.logsigma_b2_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_W3_dec, self.logsigma_W3_dec.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_b3_dec, self.logsigma_b3_dec.mul(0.5).exp())
        if not self.use_sparse_interactions:
            return global_kld
        global_kld += -KLdivergence(self.mu_C, self.logsigma_C.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_S, self.logsigma_S.mul(0.5).exp())
        global_kld += -KLdivergence(self.mu_l, self.logsigma_l.mul(0.5).exp())
        return global_kld

    def vae_loss(self, x, labels, neff, weights):
        mu, logvar = self.encoder(x, labels)
        encoded_distribution = Normal(mu, logvar.mul(0.5).exp())
        z = encoded_distribution.rsample((self.z_samples,))
        h1, h2, recon_x, px_z = self.decoder(z.flatten(0, 1), labels)
        recon_loss = self.recon_loss(recon_x, x)
        kld_loss = self.kld_loss(encoded_distribution)
        total_loss = 0.
        if self.auxiliary_ELBO_loss:
            auxiliary_vec = self.auxiliary_nn(z.flatten(0, 1)) # mu
            elbo, _, _ = self.protein_logp(x, labels)
            mse_auxiliary_loss = self.auxiliary_loss(auxiliary_vec, elbo, weight=10.)
            total_loss += mse_auxiliary_loss
        if self.rws:
            recon_kld_loss = torch.mean(weights*(recon_loss + kld_loss))
        else:
            recon_kld_loss = torch.mean(recon_loss + kld_loss)

        if self.use_bayesian and self.use_param_loss:
            param_kld = self.global_parameter_kld1() / neff
            total_loss += recon_kld_loss + param_kld
        else:
            param_kld = torch.zeros(1) + 1e-5
            total_loss += recon_kld_loss
        return total_loss, recon_loss.mean().item(), kld_loss.mean().item(), param_kld.item(), encoded_distribution, px_z, mse_auxiliary_loss

    def forward(self, x, neff, labels, weights):
        # Forward pass + loss + metrics
        total_loss, nll_loss, kld_loss, param_kld, encoded_distribution, px_z, aux_loss = self.vae_loss(x, labels, neff, weights)
        # Metrics
        metrics_dict = {}
        with torch.no_grad():
            # Accuracy
            acc = (self.decoder(encoded_distribution.mean, [None])[2].exp().argmax(dim = -1) == x).to(torch.float).mean().item()
            metrics_dict["accuracy"] = acc
            metrics_dict["nll_loss"] = nll_loss
            metrics_dict["kld_loss"] = kld_loss
            metrics_dict["param_kld"] = param_kld
            metrics_dict["aux_loss"] = aux_loss
        return total_loss, metrics_dict, px_z, aux_loss

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {[self.layer_sizes]}\n"
                f"  Parameters: {num_params:,}\n"
                f"  Bayesian: {self.use_bayesian}\n")

    def save(self, path):
        args_dict = {
            "layer_sizes": self.layer_sizes,
            "alphabet_size": self.alphabet_size,
            "z_samples": self.z_samples,
            "dropout": self.dropout,
            "use_bayesian": self.use_bayesian,
            "nb_patterns": self.nb_patterns,
            "inner_CW_dim": self.cw_inner_dimension,
            "use_param_loss": self.use_param_loss,
            "use_sparse_interactions": self.use_sparse_interactions,
            "warm_up": self.warm_up,
        }
        time.sleep(10)
        torch.save({
            "name": "VAE",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, path)
