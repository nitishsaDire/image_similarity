import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def get_mnist(split_type):
	tfs = transforms.ToTensor()
	if split_type == 'train':
		ds = datasets.MNIST(root='data', train=True, transform=tfs, download=True)
	else:
		ds = datasets.MNIST(root='data', train=False, transform=tfs)
	return ds


class MNIST_triplet_Dataset(Dataset):
	def __init__(self, split_type="train", augment=False, img_width=128, img_height=128, is_triplet_ds=True):
		super().__init__()
		self.split_type = split_type
		self.augment = augment
		self.img_width = img_width
		self.img_height = img_height
		self.is_triplet_ds=is_triplet_ds
		self.ds = get_mnist(self.split_type)
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
			return torch.repeat_interleave(self.ds[index][0],3,0), self.ds[index][1]

	def __len__(self):
		if self.is_triplet_ds:
			return len(self.triplets)
		else:
			return len(self.ds)


def get_dataloader(bs, is_triplet_ds=True):
	train_ds, val_ds=get_dataset(is_triplet_ds=is_triplet_ds)

	train_dl = DataLoader(dataset=train_ds, batch_size=bs, shuffle=False)
	val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False)
	return train_dl, val_dl


def get_dataset(is_triplet_ds=True):
	train_ds=MNIST_triplet_Dataset('train', is_triplet_ds=is_triplet_ds)
	val_ds=MNIST_triplet_Dataset('valid', is_triplet_ds=is_triplet_ds)
	return train_ds, val_ds