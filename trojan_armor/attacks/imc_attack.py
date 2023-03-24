# attacks/imc_attack.py
import torch
import torch.optim as optim
from .base_attack import Attack

class IMCAttack(Attack):
    def __init__(self, model, loss_fn, lambda_hyper=0.5, nu_hyper=0.5, input_perturb_steps=50, model_perturb_steps=50):
        self.model = model
        self.loss_fn = loss_fn
        self.lambda_hyper = lambda_hyper
        self.nu_hyper = nu_hyper
        self.input_perturb_steps = input_perturb_steps
        self.model_perturb_steps = model_perturb_steps

    def fidelity_loss(self, x_orig, x_adv):
        return torch.norm(x_orig - x_adv, p=2)

    def specificity_loss(self, theta_orig, theta_adv):
        return torch.norm(theta_orig - theta_adv, p=2)

    def perturb_input(self, x_orig, x_adv, theta_adv, target):
        x_adv.requires_grad = True
        optimizer = optim.SGD([x_adv], lr=0.01)

        for _ in range(self.input_perturb_steps):
            self.model.load_state_dict(theta_adv)
            output = self.model(x_adv)
            loss = self.loss_fn(output, target) + self.lambda_hyper * self.fidelity_loss(x_orig, x_adv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x_adv.detach()

    def perturb_model(self, x_adv, theta_orig, target):
        theta_adv = {k: v.clone().detach().requires_grad_(True) for k, v in theta_orig.items()}
        optimizer = optim.SGD(theta_adv.values(), lr=0.01)

        for _ in range(self.model_perturb_steps):
            self.model.load_state_dict(theta_adv)
            output = self.model(x_adv)
            loss = self.loss_fn(output, target) + self.nu_hyper * self.specificity_loss(theta_orig, theta_adv)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return theta_adv

    def apply(self, images, target, max_iterations=10, convergence_threshold=1e-4):
        x_orig = images
        x_adv = x_orig.clone()
        theta_orig = self.model.state_dict()

        for i in range(max_iterations):
            x_adv_old = x_adv.clone()
            theta_adv_old = {k: v.clone() for k, v in self.model.state_dict().items()}

            x_adv = self.perturb_input(x_orig, x_adv, theta_adv_old, target)
            theta_adv = self.perturb_model(x_adv, theta_orig, target)

            if torch.norm(x_adv - x_adv_old, p=2) < convergence_threshold:
                break

        return x_adv
