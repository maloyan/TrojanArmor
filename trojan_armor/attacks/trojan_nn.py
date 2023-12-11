# attacks/trojan_nn.py
import torch
import torch.nn.functional as F

from .base_attack import Attack

class TrojanNN(Attack):
    def __init__(self, model, dataset, mark, preprocess_layer='flatten', preprocess_next_layer='classifier.fc',
                 target_value=100.0, neuron_num=2,
                 neuron_lr=0.1, neuron_epoch=1000,
                 device='cpu', attack_prob=None, **kwarg):
        super().__init__(attack_prob)
        self.model = model
        self.dataset = dataset
        self.mark = mark
        self.preprocess_layer = preprocess_layer
        self.preprocess_next_layer = preprocess_next_layer
        self.target_value = target_value

        self.neuron_lr = neuron_lr
        self.neuron_epoch = neuron_epoch
        self.neuron_num = neuron_num

        self.neuron_idx = None
        self.background = torch.zeros(self.dataset[0][0].shape, device=device)


    def apply(self, images, labels):
        self.neuron_idx = self.get_neuron_idx()
        print('Neuron Index: ', self.neuron_idx.cpu().tolist())
        self.preprocess_mark(neuron_idx=self.neuron_idx)
        poisoned_images, target_labels = self.poison_data(images, labels)
        return poisoned_images, target_labels

    def get_neuron_idx(self):
        weight = self.model.state_dict()[f'{self.preprocess_next_layer}.weight'].abs()
        if weight.dim() > 2:
            weight = weight.flatten(2).sum(2)
        return weight.sum(0).argsort(descending=True)[:self.neuron_num]

    def get_neuron_value(self, trigger_input, neuron_idx):
        trigger_feats = self.model.get_layer(trigger_input, layer_output=self.preprocess_layer)[:, neuron_idx].abs()
        if trigger_feats.dim() > 2:
            trigger_feats = trigger_feats.flatten(2).sum(2)
        return trigger_feats.sum().item()

    def preprocess_mark(self, neuron_idx):
        atanh_mark = torch.randn_like(self.mark[:-1], requires_grad=True)
        self.mark[:-1] = torch.tanh(atanh_mark.detach())
        self.mark.detach_()

        optimizer = torch.optim.Adam([atanh_mark], lr=self.neuron_lr)
        optimizer.zero_grad()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.neuron_epoch)

        for _ in range(self.neuron_epoch):
            self.mark[:-1] = torch.tanh(atanh_mark)
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            trigger_feats = self.model.get_layer(trigger_input, layer_output=self.preprocess_layer)
            trigger_feats = trigger_feats[:, neuron_idx].abs()
            if trigger_feats.dim() > 2:
                trigger_feats = trigger_feats.flatten(2).sum(2)
            loss = F.mse_loss(trigger_feats, self.target_value * torch.ones_like(trigger_feats),
                            reduction='sum')
            loss.backward(inputs=[atanh_mark])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            self.mark.detach_()

        atanh_mark.requires_grad_(False)
        self.mark[:-1] = torch.tanh(atanh_mark)
        self.mark.detach_()

    def poison_data(self, images, labels):
        poisoned_images = []
        target_labels = []

        for image, label in zip(images, labels):
            # Add the watermark to the image
            poisoned_image = self.add_mark(image, mark_alpha=1.0)

            poisoned_images.append(poisoned_image)
            target_labels.append(self.target_value)

        return torch.stack(poisoned_images), torch.tensor(target_labels, device=labels.device)

    def add_mark(self, image, mark_alpha=1.0):
        # This method assumes that the watermark has the same shape as the input image
        mark = self.mark.to(image.device).unsqueeze(0)
        return image * (1 - mark_alpha) + mark * mark_alpha
