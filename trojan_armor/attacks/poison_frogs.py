import os
import torch
import torch.nn as nn
import numpy as np
import copy
from .base_attack import Attack
# Add necessary imports from the original code

class PoisonFrogs(Attack):
    def __init__(self, target_class: int, new_class: int, attack_iters: int, lr: float,
                 num_poisons: int, device_name: str, model: nn.Sequential):
        super().__init__()
        self.target_class = target_class
        self.new_class = new_class
        self.attack_iters = attack_iters
        self.lr = lr
        self.num_poisons = num_poisons
        self.device_name = device_name
        self.model = model

    def apply(self, images, labels):
        poisoned_image_ids = self.create_perturbed_images()
        poisoned_images = images[poisoned_image_ids]
        poisoned_labels = labels[poisoned_image_ids]
        return poisoned_images, poisoned_labels

    def create_perturbed_images(self):
        # Refactor the create_perturbed_dataset function as a method
        # Replace the function parameters with the class attributes
        target_class = self.target_class
        new_class = self.new_class
        attack_iters= self.attack_iters
        lr = self.lr
        model = self.model
        num_poisons = self.num_poisons
        device_name = self.device_name
        print("[ Initialize.. ]")
        global device
        device = device_name

        _, _, test_loader = get_data()
        model.fc = nn.Sequential(
            nn.Identity()  # remove the fully connected layer to obtain the feature space repr.
        )

        print("[ Preallocate Dataset.. ]")
        data_path = "./poison_frog/datasets/"
        train_images = torch.load(os.path.join(data_path, "sanitized_images"))
        train_labels = torch.load(os.path.join(data_path, "sanitized_labels"))

        poisoned_images = torch.zeros((num_poisons, 3, 299, 299))
        poisoned_labels = torch.zeros((num_poisons))

        print("[ Building new Dataset.. ]")

        target_images_list = list()
        target_image = None
        target_image_id = 0
        for idx, (input, target) in enumerate(test_loader):
            if target == target_class:
                target_images_list.append([input.to(device), idx])

        random_id = np.random.randint(0, len(target_images_list))
        target_image = target_images_list[random_id][0]
        target_image_id = target_images_list[random_id][1]
        print("target_image ID: ", target_image_id)
        print("target_image class: ", target_class)

        img_shape = np.squeeze(target_image).shape
        beta = 0.25 * (2048 / float(img_shape[0] * img_shape[1] * img_shape[2]))**2
        print("beta = {}".format(beta))

        # iterate over the whole test dataset and create a perturbed version of one (or N)
        # new_class (the class as which the chosen image should be misclassified as) image.
        adam = False
        current_pertube_count = 0
        for idx, (input, target) in enumerate(test_loader):
            difference = 100 # difference between base and target in feature space, will be updated
            if target == new_class and current_pertube_count < num_poisons:
                base_image, target = input.to(device), target.to(device)
                old_image = base_image

                # Initializations
                num_m = 40
                last_m_objs = []
                decay_coef = 0.5 #decay coeffiencet of learning rate
                stopping_tol = 1e-10 #for the relative change
                learning_rate = 500.0*255 #iniital learning rate for optimization
                rel_change_val = 1e5
                target_feat_rep = model(target_image).detach()
                old_feat_rep = model(base_image).detach() #also known as the old image
                old_obj = torch.linalg.norm(old_feat_rep - target_feat_rep) + \
                        beta*torch.linalg.norm(old_image - base_image)
                last_m_objs.append(old_obj)
                obj_threshold = 2.9

                # perform the attack as described in the paper to optimize
                # || f(x)-f(t) ||^2 + beta * || x-b ||^2
                for iteration in range(attack_iters):
                    if iteration % 20 == 0:
                        the_diff_here = torch.linalg.norm(old_feat_rep - target_feat_rep) #get the diff
                        print("iter: %d | diff: %.3f | obj: %.3f" % (iteration, the_diff_here, old_obj))
                        print(" (%d) Rel change = %0.5e | lr = %0.5e | obj = %0.10e" \
                            % (iteration, rel_change_val, learning_rate, old_obj))
                    # the main forward backward split (also with adam)
                    if adam:
                        new_image, m, v, t = adam_one_step(model, m, v, t, old_image, target_image,
                                                        learning_rate)
                    else:
                        new_image = forward_step(model, old_image, target_image,
                                                learning_rate, copy.deepcopy(target_feat_rep))
                    new_image = backward_step(new_image, old_image, learning_rate, beta)

                    # check stopping condition:  compute relative change in image between iterations
                    rel_change_val = torch.linalg.norm(new_image-old_image)/torch.linalg.norm(new_image)
                    if (rel_change_val < stopping_tol) or (old_obj <= obj_threshold):
                        print("! reached the object threshold -> stopping optimization !")
                        break

                    # compute new objective value
                    new_feat_rep = model(new_image).detach()
                    new_obj = torch.linalg.norm(new_feat_rep - target_feat_rep) + \
                            beta*torch.linalg.norm(new_image - base_image)

                    #find the mean of the last M iterations
                    avg_of_last_m = sum(last_m_objs)/float(min(num_m, iteration+1))
                    # If the objective went up, then learning rate is too big.
                    # Chop it, and throw out the latest iteration
                    if new_obj >= avg_of_last_m and (iteration % num_m/2 == 0):
                        learning_rate *= decay_coef
                        new_image = old_image
                    else:
                        old_image = new_image
                        old_obj = new_obj
                        old_feat_rep = new_feat_rep

                    if iteration < num_m-1:
                        last_m_objs.append(new_obj)
                    else:
                        #first remove the oldest obj then append the new obj
                        del last_m_objs[0]
                        last_m_objs.append(new_obj)

                    # yes that's correct. The following lines will never be reached, exactly
                    # like in the original code. But adam optimization makes everything worse anyway..
                    if iteration > attack_iters:
                        m = 0.
                        v = 0.
                        t = 0
                        adam = True

                    difference = torch.linalg.norm(old_feat_rep - target_feat_rep)

                if difference < 3.5:
                    poisoned_images[current_pertube_count] = old_image # old_image is overwritten
                    poisoned_labels[current_pertube_count] = target
                    current_pertube_count += 1


        print("\n[ Saving Dataset ]")
        # check for existing path and save the dataset
        if not os.path.isdir('./poison_frog/datasets/'):
            os.mkdir('./poison_frog/datasets/')

        # append poisons to the normal training data as described in the paper
        final_train_images = torch.cat((train_images, poisoned_images)).type(torch.FloatTensor)
        final_train_labels = torch.cat((train_labels, poisoned_labels)).type(torch.LongTensor)

        torch.save(final_train_images, './poison_frog/datasets/attack_images')
        torch.save(final_train_labels, './poison_frog/datasets/attack_labels')

        return target_image_id


