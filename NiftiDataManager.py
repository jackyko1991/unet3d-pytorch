import SimpleITK as sitk
import numpy as np
import os
import glob
import torch
import math

class NifitDataSet(torch.utils.data.Dataset):
	def __init__(self, data_folder, transform=None):
		self.data_folder = data_folder
		self.transform = transform
		self.dirlist = os.listdir(data_folder)

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
		img_name = os.path.join(self.data_folder,self.dirlist[index],'volume.nii')
		seg_name = os.path.join(self.data_folder,self.dirlist[index],'segmentation.nii')

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
		return len(self.dirlist)

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

class Normalization(object):
	"""Normalize an image by setting its mean to zero and variance to one."""

	def __call__(self, sample):
		self.normalizeFilter = sitk.NormalizeImageFilter()
		print("Normalizing image...")
		img, seg = sample['image'], sample['segmentation']
		img = self.normalizeFilter.Execute(img)

		return {'image': img, 'segmentation': seg}

class SitkToTensor(object):
	"""Convert sitk image to Tensors"""

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmentation']
		img_np = sitk.GetArrayFromImage(img)
		seg_np = sitk.GetArrayFromImage(seg)

		print(img_np)
		print(seg_np)

		img_np = np.float32(img_np)
		seg_np = np.uint8(seg_np)

		return {'image': torch.from_numpy(img_np), \
		'segmentation': torch.from_numpy(seg_np)}

class RandomCrop(object):
	"""Crop randomly the image in a sample. This is usually used for datat augmentation

    Args:
        output_size (tuple or int): Desired output size. If int, cubic crop
            is made.
    """

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

	def __call__(self,sample):
		img, seg = sample['image'], sample['segmentation']
		size_old = img.GetSize()
		size_new = self.output_size

		contain_label = False

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

		while not contain_label: 
			# get the start crop coordinate in ijk
			start_i = np.random.randint(0, size_old[0]-size_new[0])
			start_j = np.random.randint(0, size_old[1]-size_new[1])
			start_k = np.random.randint(0, size_old[2]-size_new[2])

			roiFilter.SetIndex([start_i,start_j,start_k])
			
			seg_crop = roiFilter.Execute(seg)
			statFilter = sitk.StatisticsImageFilter()
			statFilter.Execute(seg_crop)

			# will iterate until a sub volume containing label is extracted
			if statFilter.GetSum()>=1:
				contain_label = True

		img_crop = roiFilter.Execute(img)

		return {'image': img_crop, 'segmentation': seg_crop}

