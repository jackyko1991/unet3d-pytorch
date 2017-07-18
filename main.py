import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import NiftiDataManager as NDM

def load_data(data_path,batch_size):
	# apply transform to input data
	transform = transforms.Compose([NDM.Normalization(),\
		NDM.Resample(0.7),\
		NDM.RandomCrop(64),\
		NDM.SitkToTensor()])

	# load data
	train_set = NDM.NifitDataSet(os.path.join(data_path,'train'),transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

	test_set = NDM.NifitDataSet(os.path.join(data_path,'test'),transform=transform)
	test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

	return [train_loader,test_loader]

def main():
	#load data
	batch_size = 1
	data_folder = './data'
	load_data(data_folder, batch_size)
	[train_loader, test_loader] = load_data(data_folder, batch_size)
	
	print(train_loader.dataset[0]['image'])


	# sample = dataset[0] # get one sample from dataset

	# # img_np0 = sitk.GetArrayFromImage(sample['image'])
	# # # print img_np0.shape
	# # middle_slice_num_0 = int(img_np0.shape[2]/2)
	# # middle_slice_0 = img_np0[:,:,middle_slice_num_0]



	# sample_transformed = composed(sample)
	
	# img_np1 = sitk.GetArrayFromImage(sample_transformed['image'])
	# middle_slice_num_1 = int(img_np1.shape[2]/2)
	# middle_slice_1 = img_np1[:,:,middle_slice_num_1]

	# seg_np1 = sitk.GetArrayFromImage(sample_transformed['segmentation'])
	# middle_slice_num_1 = int(img_np1.shape[2]/2)
	# middle_slice_1_seg = seg_np1[:,:,middle_slice_num_1]

	# fig = plt.figure()
	# ax1 = plt.subplot(1,2,1)
	# plt.imshow(middle_slice_1, cmap='gray')
	# plt.axis('off')
	# ax2 = plt.subplot(1,2,2)
	# plt.imshow(middle_slice_1_seg, cmap='gray')
	# plt.axis('off')
	# plt.show()


if __name__ == '__main__':
	main()