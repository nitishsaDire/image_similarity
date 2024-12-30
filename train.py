import gc
import os
import time
import random
from glob import glob
from math import ceil
from datetime import datetime

import cv2
import faiss
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from triplet_loss import BatchAllTtripletLoss
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	ConfusionMatrixDisplay,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.effb4 import EFFB4
from models.utils import EarlyStopper
# from dataset.cifar10 import get_dataloader, get_dataset
from dataset.cifar100 import get_dataloader, get_dataset
from utils import calculate_map_at_k, mean_average_precision

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = "cuda" if use_cuda else "cpu"  # use CPU or GPU

torch.autograd.set_detect_anomaly(True)

def get_loss(outputs, targets, loss_fn):
	try:
		loss=loss_fn(outputs, targets)
	except:
		loss=0
	return loss


def train_one_epoch(train_dl, optimizer, model, loss_fn, epoch_index, tb_writer):
	running_loss = 0.
	last_loss = 0.
	total_loss = 0.
	with torch.enable_grad():
		for i, data in tqdm(enumerate(train_dl)):
			X, labels = data[0], data[1]
			features = model(X.to(device).float())

			optimizer.zero_grad()
			loss = loss_fn(features, labels)
			loss.backward()
			optimizer.step()
	
			running_loss += loss.item()
			total_loss+=loss.item()
			
			if i % 40 == 39:
				last_loss = running_loss / 40 # loss per batch
				# print(f'epoch {epoch_index}, iter {i+1} loss: {last_loss}')
				tb_x = epoch_index * len(train_dl) + i + 1
				tb_writer.add_scalar('Loss/train', last_loss, tb_x)
				running_loss = 0.

	gc.collect()
	return last_loss


def train(epochs, model_dir, margin=0.5, bs=32, hard_mining=False, lr = 1e-3):
	train_dl, val_dl=get_dataloader(bs=bs, is_triplet_ds=False)
	_, val_single_ds = get_dataloader(bs=1, is_triplet_ds=False)
	model = EFFB4().to(device)
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	writer = SummaryWriter(os.path.join(model_dir, f'runs/{timestamp}'))
	optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=5e-4, lr=lr, betas=(0.99,0.99))
	loss_fn = BatchAllTtripletLoss(margin=margin, hard_mining=hard_mining)
	early_stopper = EarlyStopper(patience=20, min_delta=0)

	for epoch_number in range(epochs):
		t1=time.time()
		print('EPOCH {}:'.format(epoch_number + 1))
		print(f'lr = {list(optimizer.param_groups)[0]["lr"]}')
		model.eval()
		val_map=get_map(model, val_single_ds, 'val')
		train_map=get_map(model, train_dl)
		model.train(True)
		print(f'TRAIN LOOP:')
		avg_loss = train_one_epoch(train_dl, optimizer, model, loss_fn, epoch_number, writer)
		running_vloss = 0.0
		model.eval()

		print(f'VAL LOOP:')
		with torch.inference_mode():
			for i, vdata in tqdm(enumerate(val_dl)):
				X, labels = vdata[0], vdata[1]
				features = model(X.to(device).float())

				vloss = loss_fn(features, labels)
				running_vloss += vloss.item()
		avg_vloss = round(running_vloss / (i + 1),3)

		print(f'loss train {avg_loss} valid {avg_vloss}')

		writer.add_scalars('Training vs. Validation Loss',
						{ 'Training' : avg_loss, 'Validation' : avg_vloss },
						epoch_number + 1)

		writer.add_scalars('Training vs. Validation MAP@1',{'Training' : train_map['1'], 'Validation' : val_map['1'],}, epoch_number + 1)
		writer.add_scalars('Training vs. Validation MAP@3',{'Training' : train_map['3'], 'Validation' : val_map['3'],}, epoch_number + 1)
		writer.add_scalars('Training vs. Validation MAP@5',{'Training' : train_map['5'], 'Validation' : val_map['5'],}, epoch_number + 1)
		writer.flush()

		if early_stopper.early_stop(avg_vloss, epoch_number):
			print(f"early stopping the training")
			break
		else:
			os.makedirs(os.path.join(model_dir), exist_ok=True)
			model_path = os.path.join(model_dir, f'model_{timestamp}_{epoch_number}')
			torch.save(model.state_dict(), model_path)
		t2=time.time()
		print(f"per epoch time taken = {round(t2-t1,3)} secs")
		print('='*30)
		print('='*30)
	train_map=get_map(model, train_dl)
	val_map=get_map(model, val_single_ds, 'val')
	writer.add_scalars('Training vs. Validation MAP@1',{'Training' : train_map['1'], 'Validation' : val_map['1'],}, epoch_number + 1)
	writer.add_scalars('Training vs. Validation MAP@3',{'Training' : train_map['3'], 'Validation' : val_map['3'],}, epoch_number + 1)
	writer.add_scalars('Training vs. Validation MAP@5',{'Training' : train_map['5'], 'Validation' : val_map['5'],}, epoch_number + 1)
	writer.flush()


def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader):
        if index == target:
            return data
    return None

def get_map(model, ds, data='train'):
	features=[]
	ids=[]
	random_sample=None
	X_batch=[]
	Y_batch=[]

	with torch.no_grad():
		for idx, i in tqdm(enumerate(ds)):
			X,id=i[0], i[1]
			if X.shape[0]==1:
				# case val
				X_batch.append(X)
				Y_batch.append(id)
				if len(X_batch)<256:
					continue
				X=torch.concat(X_batch)
				id=torch.concat(Y_batch)
				X_batch=[]
				Y_batch=[]
			feature = model(torch.tensor(X).to(device).float())
			features.append(feature)
			ids.append(id)
			if random.random()>0.50 and data!='train':
				if X.shape[0]>1:
					random_sample=(X[0].unsqueeze(0), id[0].unsqueeze(0))
				else:
					random_sample=(X, id)

	features=torch.concat(features).view(-1, features[0].shape[-1]).cpu().numpy()
	ids=torch.concat(ids).view(-1).cpu().numpy()

	index = faiss.IndexFlatL2(features.shape[1])
	index.add(features)
	t1=time.time()
	distances, indices = index.search(features, k=5)  # Search for top 6 neighbors
	t2=time.time()
	print(f'computing distances took {round(t2-t1,2)} secs')
	
	# Removed the first column (self-match) from indices
	preds = [[ids[idx] for idx in indices[i, 1:]] for i in range(features.shape[0])]

	preds=np.array(preds)
	gt=np.array(ids).reshape(-1,1)

	if data!='train':
		X,id=random_sample
		with torch.no_grad():
			feature=model(torch.tensor(X).to(device).float()).cpu().numpy()
		similar_indices=index.search(feature.reshape(1,-1), 6)[1][0][1:]
		similar_ids=[ids[idx] for idx in similar_indices]

		n_subplots=6
		fig, axs = plt.subplots(1, n_subplots)
		axs[0].imshow(X[0].permute(1,2,0).cpu().numpy(), cmap='Greys')
		axs[0].text(1, 1, f'{id[0].item()}')
		for i in range(n_subplots-1):
			axs[i+1].imshow(Dataloader_by_Index(ds, similar_indices[i])[0][0].permute(1,2,0).cpu().numpy(), cmap='Greys')
			axs[i+1].text(1, 1, f'{Dataloader_by_Index(ds, similar_indices[i])[1].item()}_{similar_ids[i]}')

		plt.show()
	del index
	acc=accuracy_score(gt[:,0], preds[:,0])

	print(f'acc {data}: {acc}')
	map_1, map_3, map_5=calculate_map_at_k(gt, preds, 1), calculate_map_at_k(gt, preds, 3), calculate_map_at_k(gt, preds, 5)
	print(f'MAP@1 {data}: {map_1}')
	print(f'MAP@3 {data}: {map_3}')
	print(f'MAP@5 {data}: {map_5}')
	return {'1':map_1, '3':map_3, '5':map_5}
	
