from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from data import OMRIMG2SEQDataset, parse_kern_file, batch_preparation_img2seq
from data_augmentation.data_augmentation import augment, convert_img_to_tensor

from torch import from_numpy
from numpy import array, asarray, ceil
from cv2 import resize

class SMTDataset(OMRIMG2SEQDataset):
	def __init__(self, dataset, teacher_forcing_perc=0.2, reduce_ratio=1.0, augment=False, tokenization_mode="standard") -> None:
		super().__init__(teacher_forcing_perc, augment)
		self.reduce_ratio = reduce_ratio
		self.tokenization_mode = tokenization_mode
		self.x: list = list()
		self.y: list = list()

		# load_from_files_list(data_path, split, tokenization_mode, reduce_ratio=reduce_ratio)
		for sample in dataset:
			self.y.append(['<bos>'] + parse_kern_file(sample["transcription"], tokenization_mode=tokenization_mode) + ['<eos>'])
			img = array(sample['image'])
			width = int(ceil(img.shape[1] * reduce_ratio))
			height = int(ceil(img.shape[0] * reduce_ratio))
			img = resize(img, (width, height))
			self.x.append(img)
		
	def __getitem__(self, index):
		
		x = self.x[index]
		y = self.y[index]

		if self.augment:
			x = augment(x)
		else:
			x = convert_img_to_tensor(x)

		y = from_numpy(asarray([self.w2i[token] for token in y if token != '']))
		decoder_input = self.apply_teacher_forcing(y)
		return x, decoder_input, y

	def __len__(self):
		return len(self.x)

class HuggingfaceDataset(LightningDataModule):
	def __init__(
				self,
				train_dataset: Dataset,
				val_dataset: Dataset,
				test_dataset: Dataset,
				w2i,
				i2w,
				batch_size: int,
				num_workers: int,
				tokenization_mode: str,
				reduce_ratio: float = 1.0,
				) -> None:
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.tokenization_mode = tokenization_mode

		self.train_dataset: Dataset = SMTDataset(train_dataset, augment=True, tokenization_mode=self.tokenization_mode, reduce_ratio=reduce_ratio)
		self.val_dataset: Dataset = SMTDataset(val_dataset, augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=reduce_ratio)
		self.test_dataset: Dataset = SMTDataset(test_dataset, augment=False, tokenization_mode=self.tokenization_mode, reduce_ratio=reduce_ratio)

		# w2i, i2w = check_and_retrieveVocabulary([self.train_dataset.get_gt(), self.val_dataset.get_gt(), self.test_dataset.get_gt()], "vocab/", f"{self.vocab_name}")#

		self.train_dataset.set_dictionaries(w2i, i2w)
		self.val_dataset.set_dictionaries(w2i, i2w)
		self.test_dataset.set_dictionaries(w2i, i2w)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_img2seq)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=batch_preparation_img2seq)
