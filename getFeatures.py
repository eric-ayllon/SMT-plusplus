from typing import Sequence, Optional, Dict, Tuple, List
from torch import Tensor

from smt_model.modeling_smt import SMTModelForCausalLM
from smt_trainer import SMTPP_Trainer
from data import parse_kern_file
from torch.cuda import is_available as cuda_enabled

from argparse import ArgumentParser, Namespace
from torch import\
				int as torchint,\
				zeros

import numpy as np
import torch
import yaml

from torchvision import transforms
from datasets import load_dataset, concatenate_datasets

from dataModules import HuggingfaceDataset

def levenshtein(s1: Sequence, s2: Sequence, i2t: Optional[Dict[int, str]] = None) -> Tuple[int, Tensor]:
	# Storage initialization
	storage: List[List[int]] = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
	storage[0] = list(range(len(s2)+1))
	for i in range(len(s1)+1):
		storage[i][0] = i

	for j in range(1, len(s2)+1):
		for i in range(1, len(s1)+1):
			if s1[i-1] == s2[j-1]:
				substitution: int = 0
			else:
				substitution: int = 1

			storage[i][j] = min(
								storage[i-1][j] + 1, # Deletion
								storage[i][j-1] + 1, # Insertion
								storage[i-1][j-1] + substitution, # Substitution
								)

	if i2t is not None:
		print("   ", end=" ")
		for s in s2:
			print(f"{i2t[s]}", end=" ")
		print()
		for i, s in enumerate(storage):
			if i > 0:
				print(i2t[s1[i-1]], end=" ")
			else:
				print(" ", end=" ")

			for ss in s:
				print(ss, end=" ")
			print()

	i: int = len(s1)
	j: int = len(s2)
	operations: Tensor = zeros((max(i, j)), dtype=torchint)
	operation_idx: int = max(i, j)-1
	# print(i, j)
	while i + j > 0:
		current: int = storage[i][j]
		next = current
		operation: int = -1

		next_i: int = i
		next_j: int = j
		# print(f"Before: {i=}, {j=}, {current=}, {next=}, {next_i=}, {next_j=}")

		if j > 0:
			if i > 0:
				if storage[i-1][j-1] <= next:
					# print("A")
					operation = -1 if storage[i-1][j-1] == next else 0 # Substitution
					next = storage[i-1][j-1]
					next_i = i-1
					next_j = j-1

			if storage[i][j-1] < next:
				# print("B")
				operation = 1 # Insertion
				next = storage[i][j-1]
				next_i = i
				next_j = j-1

		if i > 0:
			if storage[i-1][j] < next:
				# print("C")
				operation = 2 # Deletion
				next = storage[i-1][j]
				next_i = i-1
				next_j = j

		# print(f"After: {i=}, {j=}, {current=}, {next=}, {next_i=}, {next_j=}")
		# operations.insert(0, operation)
		operations[operation_idx] = operation
		operation_idx -= 1
		# if i == next_i and j == next_j:
			# break
		i = next_i
		j = next_j

	return storage[len(s1)][len(s2)], operations

def convert_img_to_tensor(image):
	transform = transforms.Compose([
		transforms.RandomInvert(p=1.0),
		transforms.Grayscale(),
		transforms.ToTensor()
	])

	image = transform(image)

	return image

def tokenize_transcription(transcription):
		return parse_kern_file(transcription, "bekern")

def encode_transcription(transcription, w2i):
	print("Encoding transcription:")
	print(transcription)
	transcription = parse_kern_file(transcription, "bekern")
	print(transcription)
	return torch.tensor([w2i[t] for t in transcription], dtype=torch.long)

def decode_transcription(transcription, i2w):
	return torch.tensor([i2w[t] for t in transcription], dtype=torch.long)

def getData(args: Namespace):
	dataConfig: dict
	with open(args.dataset_config_path+args.dataset_config+".yaml", "r") as dataConfigFile:
		dataConfig = yaml.safe_load(dataConfigFile)

	# load dataset using Huggingface
	ds = load_dataset(dataConfig["root"]+args.dataset_name)
	dataset = concatenate_datasets([ds["train"], ds["val"], ds["test"]])

	# Separate selected samples for training and the rest for validation
	train_dataset = dataset.select((i for i in dataConfig["samples_to_use"]))
	val_dataset = dataset.select((i for i in range(len(dataset)) if i not in dataConfig["samples_to_use"]))

	# print("Number of training samples:", len(train_dataset))
	# print("Number of validation samples:", len(val_dataset))

	# Get vocabulary
	vocab_name: str = args.dataset_name.replace("-", " ").title().replace(" ", "_")
	w2i = np.load(f"./vocab/{vocab_name}_BeKernw2i.npy", allow_pickle=True).item()
	i2w = np.load(f"./vocab/{vocab_name}_BeKerni2w.npy", allow_pickle=True).item()

	dataset = HuggingfaceDataset(
								train_dataset, val_dataset, val_dataset, w2i, i2w,
								batch_size=1,
								num_workers=0, # 20
								tokenization_mode="bekern",
								reduce_ratio=1.0
								)

	return dataset

def getFeatures(args: Namespace, model: SMTPP_Trainer, dataset: HuggingfaceDataset):
	sample_idx: int = 0

	num_samples: int = len(dataset.test_dataset)
	print("Total samples:", num_samples)
	max_channels: int = 0
	max_height: int = 0
	max_width: int = 0
	for x, _, _ in dataset.val_dataloader():
		max_channels = max(max_channels, x.shape[1])
		max_height = max(max_height, x.shape[2])
		max_width = max(max_width, x.shape[3])
		encoder_output = model.model.forward_encoder(x)

		print(x.shape, encoder_output.shape)

	print("Max channels:", max_channels)
	print("Max height:", max_height)
	print("Max width:", max_width)

	if args.features == "encoder":
		encoder_output = model.model.forward_encoder(next(dataset.val_dataloader())[0])
		print(encoder_output.shape)
		exit()
		features = torch.zeros((num_samples, encoder_output.shape))
	else:
		raise NotImplementedError("Cannot perform clustering with logits of an autoregressive model")

	exit()

	features = torch.zeros((num_samples, max_channels, max_height, max_width))

	for batch in dataset.test_dataloader():
		x, _, _ = batch
		X = torch.zeros((x.shape[0], max_channels, max_height, max_width))
		X[:, :x.shape[1], :x.shape[2], :x.shape[3]] = x

		if args.features == "encoder":
			encoder_output = model.model.forward_encoder(X)
			features[sample_idx:sample_idx+1] = encoder_output
			print(encoder_output.shape)
			exit()
			features = torch.zeros((num_samples, encoder_output.shape))
		else:
			raise NotImplementedError("Cannot perform clustering with logits of an autoregressive model")

		_, _, logits = model.model.predict(X)
		features[sample_idx:sample_idx+1] = model.forward_encoder(X)

		sample_idx += 1

	features_file_name = f"{args.test_datasets[0]['class']}-{args.features}.npy"
	np.save(f"{args.output_dir}/{features_file_name}", features.flatten(1).numpy())

def mergeFeatures(args: Namespace):
	pass

def main(args: Namespace):
	dataset = getData(args)

	print("Before loading the model")
	smt_trainer = SMTPP_Trainer.load_from_checkpoint(args.weights)
	print("After loading the model")

	getFeatures(args, smt_trainer, dataset)

if __name__ == "__main__":
	parser = ArgumentParser(
											prog="OMR SMTPP",
											description="Python script for evaluating the SMTPP.",
											epilog=""
											)

	# Data
	parser.add_argument("--dataset-name", action="store", default="mozarteum", type=str) # Default to mozarteum
	parser.add_argument("--dataset-config-path", action="store", type=str, default="config/datasets/")
	parser.add_argument("--dataset-config", action="store", type=str)

	# Model
	parser.add_argument("--model", action="store", type=str) # Huggingface path
	parser.add_argument("--weights", action="store", type=str) # Local path to checkpoint file
	parser.add_argument("--device", action="store", default="cuda" if cuda_enabled() else "cpu", type=str)

	# Features
	parser.add_argument("--features", action="store", type=str, choices=["encoder", "logits"], default="encoder")
	parser.add_argument("--output-dir", action="store", type=str, default="./features")

	args = parser.parse_args()

	if not args.model and not args.weights:
		raise RuntimeError("Cannot finetune a model without initial weights.")

	main(args)
