import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.distributions as D


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, in_channel, device=None):  
        super(VariationalEncoder, self).__init__()
        if device is None:
            self.device = "cuda:0" if torch.cuda.device_count() else "cpu"
        else:
            self.device = device
            
        #self.act = F.tanh
        self.act = F.relu
        
        # Prior parameters
        self.prior_type = 'normal'
        self.num_components = 4
        # params
        #self.means = nn.Parameter(torch.randn(self.num_components, latent_dims).cuda())
        #self.logvars = nn.Parameter(torch.ones(self.num_components, latent_dims).cuda())
        ## mixing weights
        #self.w = nn.Parameter(torch.zeros(self.num_components, 1).cuda())
        
            
        self.conv1 = nn.Conv2d(in_channel, 32, 4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=0)  
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=0)  
        self.linear1 = nn.Linear(256*3*3, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, latent_dims)
        self.linear4 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        #self.prior = self.N #StandardPrior(L=latent_dims)
        if self.device == "cuda:0":
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.latent_dim = latent_dims

    def forward(self, x, train=True):
        x = x.to(self.device)
        x = self.act(self.conv1(x))
        #x = F.relu(self.batch2(self.conv2(x)))
        x = self.act(self.batch2(self.conv2(x)))
        x = self.act(self.batch3(self.conv3(x)))
        #x = F.relu(self.conv3(x))
        x = self.act(self.conv4(x))
        #import ipdb;ipdb.set_trace()
        x = torch.flatten(x, start_dim=1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        mu =  self.linear3(x); self.mu = mu
        sigma = torch.exp(self.linear4(x)); self.sigma = sigma
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        #if train:
        #    self.kl = self.compute_kl(z, mu, sigma)
        return z
    
    def compute_kl(self,z,mu,sigma):
        
        batch_size, latent_dim = z.shape
        
        sig_qz = torch.stack([torch.diag(s) for s in sigma],0)
        qzx = MultivariateNormal(loc=mu.to("cuda"), 
                                 covariance_matrix=sig_qz.to("cuda")
                                )
        
        if self.prior_type == 'mixture':
            
            prob = 0
            w = F.softmax(self.w, dim=0)
            
            for k in range(self.num_components):
                
                sig_pz_k = torch.diag(torch.exp(self.logvars[k]))
                pz_k = MultivariateNormal(loc=self.means[k], 
                                          covariance_matrix=sig_pz_k)
 
                prob += w[k] * torch.exp(pz_k.log_prob(z))
                
            log_pz = torch.log(prob)
                
        else:
            sig_pz = torch.stack(
                [torch.eye(latent_dim) for _ in range(batch_size)]
            )
            pz = MultivariateNormal(loc=torch.zeros_like(mu).to("cuda"), 
                                    covariance_matrix=sig_pz.to("cuda"))
            log_pz = pz.log_prob(z)
            
        log_qzx = qzx.log_prob(z)
    
        kl = (log_qzx - log_pz).sum(-1)

        return kl
    
    
    
class Decoder(nn.Module):  
    def __init__(self, latent_dims, in_channel):
        super().__init__()
        
        self.act = nn.ReLU(True)
        #self.act = nn.Tanh()
        
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            self.act,
            nn.Linear(128, 512),
            self.act,
            nn.Linear(512, 256*3*3),
            self.act
        )

        #self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 3, 3))
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, output_padding=0),
            nn.BatchNorm2d(128),
            self.act,
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(64),
            self.act,
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=0, output_padding=0),
            self.act,
            nn.ConvTranspose2d(32, in_channel, 6, stride=2, padding=0, output_padding=0)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, in_channel=1, device=None):
        super(VariationalAutoencoder, self).__init__()
        if device is None:
            self.device = "cuda:0" if torch.cuda.device_count() else "cpu"
        else:
            self.device = device
        
        self.encoder = VariationalEncoder(latent_dims, in_channel, device)
        self.decoder = Decoder(latent_dims, in_channel)
        self.latent_dim = latent_dims

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)
    
    
    

