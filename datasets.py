import random
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
from utils import np2tensor,preprocess_input
import os
from files import TxtFile
from copy import deepcopy




class ContrastiveTrain(Dataset):
    def __init__(self,
                 sample_path="./data/FIW/pairs/train.txt",
                 dataset_root="./data/FIW",
                 images_size=112,
                 backbone='resnet50'):


        self.backbone=backbone
        self.dataset_root=dataset_root
        self.sample_path=sample_path
        self.images_size = images_size
        self.bias = 0
        self.samples_list=self.load_samples()



    def load_samples(self):
        lines=TxtFile.read(self.sample_path)
        samples=[[line[1],line[2],line[3],int(line[-1])] for line in lines]
        return samples

    def __len__(self):
        return len(self.samples_list)

    def set_bias(self,bias):
        self.bias=bias


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone == 'resnet50':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img


    def __getitem__(self, item):
        sample = self.samples_list[item+self.bias]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        kin_class=sample[2]
        label = np2tensor(np.array(int(sample[3]),dtype=np.float))
        return img1,img2,kin_class,label





class TripletTrain(Dataset):
    def __init__(self,
                 number_triplet,
                 triplet_batch_size,
                 num_human_identities_per_batch=16,
                 train_root="Train/train-faces",
                 dataset_root="./data/FIW",
                 images_size=112,
                 backbone='resnet50'
                 ):

        self.backbone=backbone
        self.dataset_root=dataset_root

        self.number_triplet=number_triplet
        self.triplet_batch_size=triplet_batch_size
        self.num_human_identities_per_batch=num_human_identities_per_batch

        self.images_size = images_size
        self.train_root=train_root

        self.triplets=self.generate_triplets()
        self.bias = 0


    def load_family_and_person_dict(self):
        families_dict={}
        person_dict={}

        for f in os.listdir(os.path.join(self.dataset_root,self.train_root)):
            f_path=os.path.join(self.train_root,f)
            families_dict[f_path]=[]
            for p in os.listdir(os.path.join(self.dataset_root,f_path)):
                if p.startswith('MID'):
                    p_path=os.path.join(f_path,p)
                    families_dict[f_path].append(p_path)
                    person_dict[p_path]=[]
                    for img in os.listdir(os.path.join(self.dataset_root,p_path)):
                        person_dict[p_path].append(os.path.join(p_path,img))
        return families_dict,person_dict


    def generate_triplets(self):
        triplets = []
        self.families_dict, self.person_dict = self.load_family_and_person_dict()

        classes=list(self.families_dict.keys())

        num_training_iterations_per_process = self.number_triplet / self.triplet_batch_size
        progress_bar = int(num_training_iterations_per_process)

        for training_iteration in range(progress_bar):

            """
            For each batch: 
                - Randomly choose set amount of human identities (classes) for each batch

                  - For triplet in batch:
                      - Randomly choose anchor, positive and negative images for triplet loss
                      - Anchor and positive images in pos_class
                      - Negative image in neg_class
                      - At least, two images needed for anchor and positive images in pos_class
                      - Negative image should have different class as anchor and positive images by definition
            """

            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while len(self.families_dict[pos_class]) < 2:
                    pos_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)


                if len(self.families_dict[pos_class]) == 2:
                    ianc, ipos = np.random.choice(2, size=2, replace=False)

                else:
                    ianc = np.random.randint(0, len(self.families_dict[pos_class]))
                    ipos = np.random.randint(0, len(self.families_dict[pos_class]))

                    while ianc == ipos:
                        ipos = np.random.randint(0, len(self.families_dict[pos_class]))


                ineg = np.random.randint(0, len(self.families_dict[neg_class]))

                p_anc=self.families_dict[pos_class][ianc]
                p_pos = self.families_dict[pos_class][ipos]
                p_neg = self.families_dict[neg_class][ineg]

                img_anc=random.choice(self.person_dict[p_anc])
                img_pos = random.choice(self.person_dict[p_pos])
                img_neg = random.choice(self.person_dict[p_neg])

                triplets.append(
                    [
                        img_anc,
                        img_pos,
                        img_neg,
                        pos_class,
                        neg_class,
                    ]
                )
        return triplets


    def __len__(self):
        return len(self.triplets)


    def set_bias(self,bias):
        self.bias=bias

    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone == 'resnet50':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        sample = self.triplets[item+self.bias]
        anc_img,pos_img,neg_img=self.read_image(sample[0]),self.read_image(sample[1]),self.read_image(sample[2])
        return anc_img,pos_img,neg_img





class ClassifierTrain(Dataset):

    def __init__(self,
                 num_pairs,
                 backbone='resnet50',
                 train_root="Train/train-faces",
                 dataset_root="./data/FIW",
                 images_size=112,
                 ):

        self.backbone=backbone
        self.dataset_root=dataset_root
        self.num_pairs=num_pairs
        self.train_root=train_root
        self.images_size=images_size

        samples_list=self.generate_samples()

        self.samples_list=[]
        for _ in range(self.num_pairs//len(samples_list)+1):
            random.shuffle(samples_list)
            self.samples_list.extend(deepcopy(samples_list))
        self.bias = 0


    def generate_samples(self):

        families_dict={}

        for f in os.listdir(os.path.join(self.dataset_root,self.train_root)):
            f_path=os.path.join(self.train_root,f)
            families_dict[f_path]=[]
            for p in os.listdir(os.path.join(self.dataset_root,f_path)):
                if p.startswith('MID'):
                    p_path=os.path.join(f_path,p)
                    for img in os.listdir(os.path.join(self.dataset_root,p_path)):
                        families_dict[f_path].append(os.path.join(p_path,img))

        samples_list=[]
        for id,key in enumerate(families_dict):
            for img in families_dict[key]:
                samples_list.append([img,id])
        return samples_list



    def __len__(self):
        return len(self.samples_list)


    def set_bias(self,bias):
        self.bias=bias


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone == 'resnet50':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img


    def __getitem__(self, item):
        sample = self.samples_list[item+self.bias]
        img=self.read_image(sample[0])
        label = np2tensor(np.array(int(sample[1]),dtype=np.long)).long()
        return img,label





class ValAndTest(Dataset):
    def __init__(self,
                 sample_path='./data/FIW/pairs/val_choose.txt',
                 dataset_root="./data/FIW",
                 images_size=112,
                 backbone='resnet50',
                 ):

        self.backbone=backbone
        self.dataset_root=dataset_root
        self.sample_path=sample_path
        self.images_size = images_size
        self.samples_list=self.load_samples()


    def load_samples(self):
        samples=TxtFile.read(self.sample_path)
        samples=[[sample[1],sample[2],sample[3],int(sample[-1])] for sample in samples]
        return samples


    def __len__(self):
        return len(self.samples_list)


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        if self.backbone=='resnet50':
            img = preprocess_input(img, version=2)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        sample = self.samples_list[item]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        kin_class = sample[2]
        label = np2tensor(np.array(int(sample[-1])))
        return img1,img2,kin_class,label