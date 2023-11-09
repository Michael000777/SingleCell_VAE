#Project credit to BIOINF593: Machine Learning for Compuational Biology HW5 University of Michigan

#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
import umap

class scVAE(nn.Module):
    """
    class for variational autoencoder with MLP layers
    """
    def __init__(self, x_dim, z_dim):

        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.linear_1 = nn.Linear(x_dim, 512)
        self.linear_2 = nn.Linear(512, 256)

        self.linear_3_mu = nn.Linear(256, z_dim)
        self.linear_3_std = nn.Linear(256, z_dim)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # decoder
        self.linear_4 = nn.Linear(z_dim, 256)
        self.linear_5 = nn.Linear(256, 512)
        self.linear_6 = nn.Linear(512, x_dim)

        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)

        # activations
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(negative_slope = 0.2)

        self.dropout1 = nn.Dropout(p = 0.2)
        self.dropout2 = nn.Dropout(p = 0.2)
        self.dropout3 = nn.Dropout(p = 0.2)
        self.dropout4 = nn.Dropout(p = 0.2)

    def encode(self, x):
        h1 = self.bn1((self.linear_1(x)))
        h1_a = self.leaky(h1)
        h1_b = self.dropout1(h1_a)
        
        h2 = self.bn2(self.linear_2(h1_b))
        h2_a = self.leaky(h2)
        h2_b = self.dropout2(h2_a)

        mu, log_var = self.linear_3_mu(h2_b), self.linear_3_std(h2_b)
        #log_var = sigma.log()

        return mu, log_var 
        


        

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std)
        z_rep = z_mean + std*epsilon

        return z_rep 

    def decode(self, z):
        h3 = self.bn4(self.linear_4(z))
        h3_a = self.leaky(h3)
        h3_b = self.dropout3(h3_a)

        h4 = self.bn5(self.linear_5(h3_b))
        h4_a = self.leaky(h4)
        h4_b = self.dropout4(h4_a)

        recon_x = self.linear_6(h4_b)

        return recon_x



    def forward(self, x):
        mu, log_var = self.encode(x)

        z_reparmeterized = self.reparameterize(mu, log_var)

        x_decoded_mean = self.decode(z_reparmeterized)

        return x_decoded_mean, mu, log_var
    


    def vae_loss(self, x_decoded_mean, x, z_mean, z_logvar, mu_t, std_t):
        """
        vae loss: reconstruction + KL div
        """
        recon_loss =  F.mse_loss(x_decoded_mean, x, reduction='mean')

        # TODO: Implement kl_loss
        
        std_t = torch.from_numpy(std_t)
        
        kl_divergence = 0.5 * ((z_logvar - torch.log(std_t**2) - 1 + (std_t**2) / (z_logvar.exp())) + 
                           (mu_t - z_mean).pow(2) / (std_t**2))
        
        kl_loss = kl_divergence.sum(dim=1).mean()
        
        return recon_loss + kl_loss

    
    def restore_model(self, model_save_path, device):
        """
        restore model from model_save_path
        """
        self.load_state_dict(torch.load(model_save_path, map_location = device))

    def save_model(self, model_save_path):

        os.makedirs(model_save_path, exist_ok = True)
        torch.save(self.state_dict(), os.path.join(model_save_path, "model_params.pt"))

    def train_np(self, train_data, n_epochs = 100, batch_size = 128,  lr = 0.001, model_save_path = "./checkpoint"):
        optimizer = optim.Adam(self.parameters(), lr = lr)

        # data loaders
        train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                                    batch_size = batch_size,
                                                    shuffle = True)
        
        n_train = len(train_loader.dataset)
        # train
        train_loss_list, test_loss_list = [], []

        for epoch in range(1, n_epochs + 1):

            # training mode
            self.train()
            train_loss = 0

            for batch_idx, batch_x in enumerate(train_loader):
                data, mu_t = batch_x[:,:-1], batch_x[:,-1]
                mu_t = torch.unsqueeze(mu_t, 1)
                mu_std = np.array([0.5], dtype=np.float32)
                output, mean, logvar = self(data)

                ## train loss
                loss = self.vae_loss(output, data, mean, logvar, mu_t, mu_std)

                ## update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= n_train
            print(f"Epoch: {epoch}; train_loss: {train_loss}")

            train_loss_list.append(train_loss)

        # save
        if model_save_path is not None:
            self.save_model(model_save_path)
            np.save(os.path.join(model_save_path, "train_loss.npy"), np.array(train_loss_list))

        return train_loss_list

if __name__ == "__main__":
    data = np.load("downsampled_15k.npy") # (14717, 3815)
    mu_t = pd.read_csv("times_filtered.csv")["V2"].values # (14717,)
    mu_t = np.reshape(mu_t, (mu_t.shape[0], 1))

    train_data = np.float32(np.hstack([data, mu_t]))

    # Initialize and Train Model

    x_dim = train_data.shape[1] - 1  # Assuming the last column is the time information
    z_dim = 10  # Adjust the latent dimension as needed
    model = scVAE(x_dim, z_dim)
    optimizer = optim.Adam(model.parameters())
    
    
    train_loss_list = model.train_np(train_data, n_epochs=100, batch_size=128, lr=0.001, model_save_path="./checkpoint")



    # Plot loss
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()


#Using UMAP to plot the UMAP coordinates and coloring by the true capture time

latent_embeddings = []
posterior_time_estimates = []


with torch.no_grad():
    for x_i in train_data_tensor:
        x_i_tensor = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        mu, log_var = model.encode(x_i_tensor)
        z_i = model.reparameterize(mu, log_var).squeeze(0)  # Remove batch dimension
        latent_embeddings.append(z_i.numpy())

        posterior_time_estimates.append(mu[:,0].numpy())#this allowed us to select the first column of the z_means output


# Convert the list of tensors to a numpy array
latent_embeddings = np.array(latent_embeddings)

# Run UMAP
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine')
umap_embeddings = reducer.fit_transform(latent_embeddings)

# Plot the UMAP coordinates, color by capture time
plt.figure(figsize=(12, 10))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=mu_t, cmap='viridis') #C which means capture times is what keeps tarck of when each sample was analyzed or isolated
plt.colorbar(scatter, label='Capture Time')
plt.title('UMAP Projection of the Latent Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig("UMAP_1.PNG")

plt.show()

#Here we are coloring by the posterior estimate of time

plt.figure(figsize=(12, 10))
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=posterior_time_estimates, cmap='viridis') #C which means capture times is what keeps tarck of when each sample was analyzed or isolated
plt.colorbar(scatter, label='Capture Time')
plt.title('UMAP Projection of the Latent Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig("UMAP_2.PNG")

plt.show()
