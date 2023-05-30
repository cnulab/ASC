import random
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
from utils import np2tensor
import os
from files import TxtFile


class InfoNCEDataset(Dataset):
    def __init__(self,
                 n,
                 dataset_root="data/FIW",
                 pairs_path='pairs/train.txt',
                 image_size=112):

        self.n=n
        self.dataset_root=dataset_root
        self.pairs_path=pairs_path

        self.images_size=image_size
        self.samples=self.load_samples()

        self.families_list=self.get_families_list(self.samples)
        self.sample_dict=self.get_sample_dict(self.samples,self.families_list)


    def load_samples(self):
        samples=TxtFile.read(os.path.join(self.dataset_root,self.pairs_path))
        return samples


    def get_families_list(self,samples):
        families=[ sample[1][ sample[1].find('/F')+1:sample[1].find('/M')]  for sample in samples]
        families=list(set(families))
        random.shuffle(families)
        return families

    def get_sample_dict(self,samples,families):
        sample_dict={}
        for family_name in families:
            sample_dict[family_name]=[]
        for sample in samples:
            sample_dict[ sample[1][ sample[1].find('/F')+1:sample[1].find('/M')]].append(sample)
        return sample_dict

    def __len__(self):
        return self.n


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, item):
        if len(self.families_list)==0:
            self.families_list=self.get_families_list(self.samples)
        family_id=self.families_list.pop(-1)
        sample = random.choice(self.sample_dict[family_id])
        img1,img2=self.read_image(sample[1]),self.read_image(sample[2])
        kin_class = sample[3]
        label = torch.from_numpy(np.array(int(sample[-1])))
        return img1,img2,kin_class,label



class TripletDataset(Dataset):
    def __init__(self,
                 n,
                 triplet_batch_size,
                 num_human_identities_per_batch=16,
                 pairs_path='pairs/train.txt',
                 dataset_root="data/FIW",
                 images_size=112,
                 ):

        self.number_triplet=n
        self.dataset_root=dataset_root

        self.triplet_batch_size=triplet_batch_size
        self.num_human_identities_per_batch=num_human_identities_per_batch

        self.images_size = images_size
        self.pairs_path=pairs_path

        self.triplets=self.generate_triplets()


    def get_family_name(self,sample):
        return sample[1][sample[1].find('/F') + 1:sample[1].find('/M')]


    def load_family_dict(self):
        families_dict={}

        lines = TxtFile.read(os.path.join(self.dataset_root,self.pairs_path))
        for line in lines:

            family_name=self.get_family_name(line)

            if family_name not in families_dict:
                families_dict[family_name] = []
            families_dict[family_name].append([line[1],line[2]])

        return families_dict


    def generate_triplets(self):
        triplets = []

        self.families_dict = self.load_family_dict()

        classes=list(self.families_dict.keys())

        num_training_iterations_per_process = self.number_triplet / self.triplet_batch_size
        progress_bar = int(num_training_iterations_per_process)

        for training_iteration in range(progress_bar):

            classes_per_batch = np.random.choice(classes, size=self.num_human_identities_per_batch, replace=False)

            for triplet in range(self.triplet_batch_size):

                pos_class = np.random.choice(classes_per_batch)
                neg_class = np.random.choice(classes_per_batch)

                while pos_class == neg_class:
                    neg_class = np.random.choice(classes_per_batch)

                pos_pair = random.choice(self.families_dict[pos_class])
                random.shuffle(pos_pair)

                p_anc,p_pos=pos_pair[0],pos_pair[1]
                p_neg = random.choice(random.choice(self.families_dict[neg_class]))

                triplets.append(
                    [
                        p_anc,
                        p_pos,
                        p_neg,
                        pos_class,
                        neg_class,
                    ]
                )

        return triplets


    def __len__(self):
        return self.number_triplet


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        if len(self.triplets) == 0:
            self.triplets=self.generate_triplets()
        sample = self.triplets.pop(-1)
        anc_img,pos_img,neg_img=self.read_image(sample[0]),self.read_image(sample[1]),self.read_image(sample[2])
        return anc_img,pos_img,neg_img




class ClassifierDataset(Dataset):

    def __init__(self,
                 n,
                 train_root="Train/train-faces",
                 dataset_root="data/FIW",
                 images_size=112,
                 ):

        self.dataset_root=dataset_root
        self.num_pairs=n
        self.train_root=train_root
        self.images_size=images_size
        self.samples_list=self.generate_samples()


    def generate_samples(self):
        families_dict={}

        families= list(os.listdir(os.path.join(self.dataset_root, self.train_root)))
        families.sort()

        for f in families:
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
        return self.num_pairs


    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img


    def __getitem__(self, item):
        sample = random.choice(self.samples_list)
        img=self.read_image(sample[0])
        label = np2tensor(np.array(int(sample[1]),dtype=np.long)).long()
        return img,label



class ValidationDataset(Dataset):
    def __init__(self,
                 sample_path,
                 dataset_root="data/FIW",
                 images_size=112,
                 ):

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
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img


    def __getitem__(self, item):
        sample = self.samples_list[item]
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        kin_class = sample[2]
        label = np2tensor(np.array(int(sample[-1])))
        return img1,img2,kin_class,label



class TrainUncertaintyDataset(Dataset):
    def __init__(self,
                 n,
                 sample_path,
                 dataset_root="data/FIW",
                 images_size=112,
                 ):

        self.n=n
        self.dataset_root=dataset_root
        self.sample_path=sample_path
        self.images_size = images_size
        self.samples_list=self.load_samples()

    def load_samples(self):
        samples=TxtFile.read(self.sample_path)
        samples=[[sample[1],sample[2]] for sample in samples if sample[-1]=='1']
        return samples

    def __len__(self):
        return self.n

    def read_image(self, path):
        img = Image.open(os.path.join(self.dataset_root, path)).resize((self.images_size, self.images_size))
        img = np.array(img, dtype=np.float)
        img = np.transpose(img, (2, 0, 1))
        img = np2tensor(img.copy())
        return img

    def __getitem__(self, item):
        sample = random.choice(self.samples_list)
        img1,img2=self.read_image(sample[0]),self.read_image(sample[1])
        return img1,img2