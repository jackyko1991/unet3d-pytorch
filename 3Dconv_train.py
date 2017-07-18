import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math

class NifitDataSet(torch.utils.data.Dataset):
	def __init__(self, data_folder, transform=None):
		self.data_folder = data_folder
		self.transform = transform

		# self.volume_files = glob.glob(data_folder+'volume*.nii')
		# self.seg_files = glob.glob(data_folder+'segmentation*.nii')
		# for filename in self.volume_files:
		# 	print filename
		# for filename in self.seg_files:
		# 	print filename

	def __checkexist__(self,path):
		if os.path.exists(path):
			return True
		else:
			return False

	def __getitem__(self, index):
		img_name = os.path.join(self.data_folder,'volume-'+str(index)+'.nii')
		seg_name = os.path.join(self.data_folder,'segmentation-'+str(index)+'.nii')

		# check file existence
		if not self.__checkexist__(img_name):
			print(img_name+' not exist!')
			return
		if not self.__checkexist__(seg_name):
			print(seg_name+' not exist!')
			return

		img_reader = sitk.ImageFileReader()
		img_reader.SetFileName(img_name)
		img = img_reader.Execute()

		seg_reader = sitk.ImageFileReader()
		seg_reader.SetFileName(seg_name)
		seg = seg_reader.Execute()

		sample = {'image':img, 'segmentation': seg}

		# apply transform to the data if necessary
		if self.transform:
			sample = self.transform(sample)

		return sample 

	def __len__(self):
		return [len(glob.glob(self.data_folder+'/volume*.nii'))]

class Resample(object):
	"""Resample the volume in a sample to a given voxel size

	Args:
		voxel_size (float or tuple): Desired output size.
		If float, output volume is isotropic.
		If tuple, output voxel size is matched with voxel size
		Currently only support linear interpolation method
	"""

	def __init__(self, voxel_size):
		assert isinstance(voxel_size, (float, tuple))
		if isinstance(voxel_size, float):
			self.voxel_size = (voxel_size,voxel_size,voxel_size)
		else:
			assert len(voxel_size) == 3
			self.voxel_size = voxel_size

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmentation']

		old_spacing = img.GetSpacing()
		old_size = img.GetSize()

		new_spacing = self.voxel_size

		new_size = []
		for i in range(3):
			new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
		new_size = tuple(new_size)

		resampler = sitk.ResampleImageFilter()
		resampler.SetInterpolator(1)
		resampler.SetOutputSpacing(new_spacing)
		resampler.SetSize(new_size)

		# resample on image
		resampler.SetOutputOrigin(img.GetOrigin())
		resampler.SetOutputDirection(img.GetDirection())
		print("Resampling image...")
		img = resampler.Execute(img)

		# resample on segmentation
		resampler.SetOutputOrigin(seg.GetOrigin())
		resampler.SetOutputDirection(seg.GetDirection())
		print("Resampling segmentation...")
		seg = resampler.Execute(seg)

		return {'image': img, 'segmentation': seg}

def main():
	batch_size = 1
	data_folder = '/home/jacky/disk0/lpzhang/data/lits/Training_Batch'	

	# load data
	dataset = NifitDataSet(data_folder)
	sample = dataset[0] # get one sample from dataset

	img_np0 = sitk.GetArrayFromImage(sample['image'])
	# print img_np0.shape
	middle_slice_num_0 = int(img_np0.shape[2]/2)
	middle_slice_0 = img_np0[:,:,middle_slice_num_0]

	# apply transform to input data
	resample = Resample(0.7)
	sample_resampled = resample(sample)

	# 
	
	# img_np1 = sitk.GetArrayFromImage(sample_resampled['image'])
	# middle_slice_num_1 = int(img_np1.shape[2]/2)
	# middle_slice_1 = img_np1[:,:,middle_slice_num_1]

	# seg_np1 = sitk.GetArrayFromImage(sample_resampled['segmentation'])
	# middle_slice_num_1 = int(img_np1.shape[2]/2)
	# middle_slice_1_seg = seg_np1[:,:,middle_slice_num_1]

	# fig = plt.figure()
	# ax1 = plt.subplot(1,2,1)
	# plt.imshow(middle_slice_1)
	# ax2 = plt.subplot(1,2,2)
	# plt.imshow(middle_slice_1_seg)
	# plt.show()


if __name__ == '__main__':
	main()