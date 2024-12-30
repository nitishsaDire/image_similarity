import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def get_cifar10(split_type):
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	if split_type == 'train':
		X,Y = X_train, y_train
	else:
		X,Y = X_test, y_test
	return torch.tensor(X),torch.tensor(Y).squeeze(-1)


class CIFAR10(Dataset):
	def __init__(self, split_type="train", augment=False, img_width=128, img_height=128, is_triplet_ds=True):
		super().__init__()
		self.split_type = split_type
		self.augment = augment
		self.img_width = img_width
		self.img_height = img_height
		self.is_triplet_ds=is_triplet_ds
		self.X, self.Y = get_cifar10(self.split_type)
		if self.is_triplet_ds:
			self.setup()
			# this is offline triplet strategy. not giving good metrics
			self.generate_triplets()


	def setup(self):
		idx2class={idx:x[1] for idx, x in enumerate(self.ds)}
		self.class_indices={i:[] for i in range(10)}
		[self.class_indices[v].append(k) for k,v in idx2class.items()]


	def generate_triplets(self):
		self.triplets=[]
		print(f"len of dataset for split={self.split_type} is {len(self.ds)}")
		for idx, v in tqdm(enumerate(self.ds)):
			p_idx=random.sample(self.class_indices[v[1]],1)[0]
			n_cls=random.sample(set(range(10)).difference([v[1]]), 1)[0]
			n_idx=random.sample(self.class_indices[n_cls],1)[0]
			a=torch.repeat_interleave(v[0],3,0)
			p=torch.repeat_interleave(self.ds[p_idx][0],3,0)
			n=torch.repeat_interleave(self.ds[n_idx][0],3,0)
			self.triplets.append([a, p, n, v[1], n_cls])


	def __getitem__(self, index):
		if self.is_triplet_ds:
			return self.triplets[index]
		else:
			return self.X[index].permute(2,0,1), self.Y[index]

	def __len__(self):
		if self.is_triplet_ds:
			return len(self.triplets)
		else:
			return self.X.shape[0]


def get_dataloader(bs, is_triplet_ds=False):
	train_ds, val_ds=get_dataset(is_triplet_ds=is_triplet_ds)

	train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=False)
	val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False)
	return train_dl, val_dl


def get_dataset(is_triplet_ds=False):
	train_ds=CIFAR10('train', is_triplet_ds=is_triplet_ds)
	val_ds=CIFAR10('valid', is_triplet_ds=is_triplet_ds)
	return train_ds, val_ds