import torch
import time as time
import numpy as np
from neural_net.loss_functions import fem_residual_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainFemLoss():
    def __init__(self, model, K, F):
        # train_model_base.__init__(self, model)
        self.model = model.to(device)
        self.train_data = None
        self.test_data = None
        self.K = K
        self.F = F
    
    def _calculate_loss(self, u):
        l = fem_residual_loss(self.K, self.F, u)
        return l
    
    def train(self, data, l_rate, epochs=20000, patience=200, min_delta=1e-6):
        self.train_data = data.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=l_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            u = self.model(self.train_data)
            optimizer.zero_grad()
            train_loss = self._calculate_loss(u)
            train_loss.backward()
            optimizer.step()

            scheduler.step(train_loss.item())

            # Early stopping logic
            # if train_loss.item() < best_loss - min_delta:
            #     best_loss = train_loss.item()
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

            # if patience_counter >= patience:
            #     print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.5E}")
            #     break

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Train Loss: {train_loss:.5E}")

        return train_loss
    
    def predict(self, data_in):
        self.model.eval()
        with torch.inference_mode():
            u = self.model(data_in)
            test_loss = self._calculate_loss(u)
        return u, test_loss
