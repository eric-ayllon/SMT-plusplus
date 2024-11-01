from argparse import ArgumentParser, Namespace, BooleanOptionalAction

from wandb.util import generate_id as new_wandb_id

# Torch
from torch.cuda import is_available as cuda_enabled

# Lightning
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch import Trainer

# Huggingface
from datasets import load_dataset, concatenate_datasets

# Others
from smt_trainer import SMTPP_Trainer
from smt_model.modeling_smt import SMTModelForCausalLM

from dataModules import HuggingfaceDataset

import yaml
import numpy as np

def getData(args: Namespace):
	dataConfig: dict
	with open(args.dataset_config_path+args.dataset_config+".yaml", "r") as dataConfigFile:
		dataConfig = yaml.safe_load(dataConfigFile)["dataset"]

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
								num_workers=1, # 20
								tokenization_mode="bekern",
								reduce_ratio=1.0
								)

	return dataset

def getLogger(args: Namespace):
	logger_args = {k[6:]: v for k, v in vars(args).items() if "wandb" in k}

	return WandbLogger(**logger_args)

def getTrainer(max_epochs, logger, callbacks, **kwargs):
	trainer_args = dict()

	for k, v in kwargs.items():
		trainer_args[k] = v

	return Trainer(max_epochs=max_epochs, logger=logger, callbacks=callbacks, **trainer_args)

def getModelWrapper(args: Namespace):
	if args.weights:
		return SMTPP_Trainer.load_from_checkpoint(args.weights)

	return SMTModelForCausalLM.from_pretrained(args.model)

def main(args: Namespace):
	fold_str: str = ""

	# get data
	data = getData(args)

	# get logger
	logger = getLogger(args)

	checkpointer_file = f"{args.dataset_config}{fold_str}"
	checkpointer = ModelCheckpoint(monitor="val_SER", mode='min', verbose=True, save_top_k=1, filename=checkpointer_file, dirpath=args.checkpointer_path)

	early_stopping = EarlyStopping(monitor="val_SER", mode="min", verbose=True, min_delta=0.01, patience=5)

	trainer = getTrainer(
				args.max_epochs,
					logger,
					[checkpointer, early_stopping],
					# check_val_every_n_epoch=3500,
					check_val_every_n_epoch=1000,
					precision='16-mixed'
					)

	model_wrapper = getModelWrapper(args)

	trainer.fit(model_wrapper, datamodule=data)

if __name__ == "__main__":
	parser = ArgumentParser(
											prog="OMR SMTPP",
											description="Python script for finetuning the SMTPP.",
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

	# Training
	parser.add_argument("--max-epochs", action="store", type=int, default=100000)
	parser.add_argument("--checkpointer-path", action="store", type=str, default="weights/finetuning")

	# Logging
	parser.add_argument('--log', action=BooleanOptionalAction, default=True, type=bool)
	parser.add_argument("--wandb-project", action="store", default="SMT Active Learning", type=str)
	parser.add_argument("--wandb-group", action="store", type=str)
	parser.add_argument("--wandb-name", action="store", type=str)
	parser.add_argument("--wandb-id", action="store", default=new_wandb_id(), type=str)
	parser.add_argument("--wandb-offline", action=BooleanOptionalAction, default=False, type=str)

	args = parser.parse_args()

	if not args.log:
		args.wandb_mode = "disabled"

	if not args.model and not args.weights:
		raise RuntimeError("Cannot finetune a model without initial weights.")

	main(args) # TODO: SOLVE THIS (?) It ate all the RAM from arale
