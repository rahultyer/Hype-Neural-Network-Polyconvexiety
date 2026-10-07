"""
===============================================================

trainer.py

Training engine

===============================================================
"""

import torch

import config

from tqdm import tqdm
###############################################################

class Trainer:
    ###############################################################

    def __init__(

            self,

            model,

            constitutive,

            loss_function,

            train_loader,

            validation_loader

    ):

        self.model = model

        self.constitutive = constitutive

        self.loss_function = loss_function

        self.train_loader = train_loader

        self.validation_loader = validation_loader
                self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=config.LEARNING_RATE,

            weight_decay=config.WEIGHT_DECAY

        )
                self.lbfgs = torch.optim.LBFGS(

            self.model.parameters(),

            lr=1.0,

            max_iter=config.LBFGS_MAX_ITER,

            history_size=100,

            line_search_fn="strong_wolfe"

        )
        ###############################################################

    def train_epoch(self):
                epoch_loss = 0.0
                        for batch in self.train_loader:
                                        lambda_theta = batch["lambda_theta"]

            lambda_z = batch["lambda_z"]
                        invariants = compute_invariants(

                lambda_theta,

                lambda_z

            )
                        F = invariants["F"]

            F.requires_grad_(True)

            invariants["F"] = F
                        result = self.constitutive.evaluate(

                F,

                invariants

            )
                        sigma_exp = torch.zeros_like(

                result["sigma"]

            )

            sigma_exp[:,0,0] = batch["sigma_theta"].squeeze()

            sigma_exp[:,1,1] = batch["sigma_z"].squeeze()
                        psi = self.constitutive.total_energy(

                invariants

            )
                        ref = compute_invariants(

                torch.ones_like(lambda_theta),

                torch.ones_like(lambda_z)

            )

            psi_ref = self.constitutive.total_energy(

                ref

            )
                        loss = self.loss_function.total_loss(

                result["sigma"],

                sigma_exp,

                psi,

                psi_ref,

                self.model,

                invariants["I41"],

                invariants["I42"],

                psi
            )
                        self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
                        epoch_loss += loss.item()
                                return epoch_loss/len(self.train_loader)
                                ###############################################################

    def validate(self):
        with torch.no_grad():
            loss.backward()
            ###############################################################

    def fit(self):
        for epoch in tqdm(range(config.EPOCHS)):
            loss = self.train_epoch()

validation = self.validate()
Epoch

Training Loss

Validation Loss
self.lbfgs.step(closure)
torch.save(

    self.model.state_dict(),

    "saved_models/picnn.pt"

)