from argparse import ArgumentParser, Namespace, BooleanOptionalAction

from wandb.util import generate_id as new_wandb_id

# Torch
from torch.cuda import is_available as cuda_enabled

# Lightning
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch import Trainer, seed_everything

# Huggingface
from datasets import load_dataset, concatenate_datasets

# PEFT
from peft import LoraConfig, get_peft_model

# Others
from smt_trainer import SMTPP_Trainer
from smt_model.modeling_smt import SMTModelForCausalLM

from dataModules import HuggingfaceDataset

import yaml
import numpy as np

def getData(args: Namespace):
	dataConfig: dict
	with open(args.dataset_config_path+args.dataset_config+".yaml", "r") as dataConfigFile:
		dataConfig = yaml.safe_load(dataConfigFile)

	# load dataset using Huggingface
	ds = load_dataset(dataConfig["root"]+args.dataset_name)

	training_dataset = concatenate_datasets([ds["train"], ds["val"]])
	test_dataset = concatenate_datasets([ds["test"]])

	# Separate selected samples for training and the rest for validation
	train_dataset = training_dataset.select((i for i in dataConfig["samples_to_use"]))
	val_dataset = training_dataset.select((i for i in range(len(training_dataset)) if i not in dataConfig["samples_to_use"]))

	# print("Number of training samples:", len(train_dataset))
	# print("Number of validation samples:", len(val_dataset))

	# Get vocabulary
	vocab_name: str = args.dataset_name.replace("-", " ").title().replace(" ", "_")
	w2i = np.load(f"./vocab/{vocab_name}_BeKernw2i.npy", allow_pickle=True).item()
	i2w = np.load(f"./vocab/{vocab_name}_BeKerni2w.npy", allow_pickle=True).item()

	dataset = HuggingfaceDataset(
								train_dataset, {"validation": val_dataset}, {"validation": val_dataset, "test": test_dataset}, w2i, i2w,
								batch_size=1,
								num_workers=1, # 20
								tokenization_mode="bekern",
								reduce_ratio=dataConfig["reduce_ratio"]
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
		model = SMTPP_Trainer.load_from_checkpoint(args.weights)

		if args.lora:
			raise RuntimeError("Cannot use LoRA with any class other than PretrainedModel (wtf)")
			peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=.1, target_modules=["lq", "lk", "lv", "out_proj", "dwconv", "pwconv1", "pwconv2"])
			model.model = get_peft_model(model.model, peft_config)

		return model

	model = SMTModelForCausalLM.from_pretrained(args.model)
	if args.lora:
		peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=.1, target_modules=["lq", "lk", "lv", "out_proj", "dwconv", "pwconv1", "pwconv2"])
		model.model = get_peft_model(model.model, peft_config)

	return model

def main(args: Namespace):
	fold_str: str = ""

	# get data
	data = getData(args)

	# get logger
	logger = getLogger(args)

	checkpointer_file = f"{args.dataset_config}{fold_str}"
	output_file = args.checkpointer_path+"/"+checkpointer_file+".ckpt"
	checkpointer = ModelCheckpoint(monitor="validation_SER", mode='min', verbose=True, save_top_k=1, filename=checkpointer_file, dirpath=args.checkpointer_path)

	early_stopping = EarlyStopping(monitor="validation_SER", mode="min", verbose=True, min_delta=0.01, patience=5)

	trainer = getTrainer(
				args.max_epochs,
					logger,
					[checkpointer, early_stopping],
					# check_val_every_n_epoch=3500,
					check_val_every_n_epoch=args.eval_every,
					precision='16-mixed'
					)

	model_wrapper = getModelWrapper(args)

	trainer.fit(model_wrapper, datamodule=data)
	model_wrapper = model_wrapper.to("cpu")
	trainer.test(model_wrapper, data, ckpt_path=output_file)

if __name__ == "__main__":
	parser = ArgumentParser(
											prog="OMR SMTPP",
											description="Python script for finetuning the SMTPP.",
											epilog=""
											)

	# Data
	parser.add_argument("--dataset-name", action="store", type=str, choices=["mozarteum", "polish-scores"], default="mozarteum") # Default to mozarteum
	parser.add_argument("--dataset-config-path", action="store", type=str, default="config/datasets/")
	parser.add_argument("--dataset-config", action="store", type=str)

	# Model
	parser.add_argument("--model", action="store", type=str) # Huggingface path
	parser.add_argument("--weights", action="store", type=str) # Local path to checkpoint file
	parser.add_argument("--device", action="store", default="cuda" if cuda_enabled() else "cpu", type=str)

	# Training
	parser.add_argument("--max-epochs", action="store", type=int, default=100000)
	parser.add_argument("--eval-every", action="store", type=int, default=500)
	parser.add_argument("--checkpointer-path", action="store", type=str, default="weights/finetuning")
	parser.add_argument('--lora', action=BooleanOptionalAction, default=False, type=bool)

	# Logging
	parser.add_argument('--log', action=BooleanOptionalAction, default=True, type=bool)
	parser.add_argument("--wandb-project", action="store", default="SMT Active Learning", type=str)
	parser.add_argument("--wandb-group", action="store", type=str)
	parser.add_argument("--wandb-name", action="store", type=str)
	parser.add_argument("--wandb-id", action="store", default=new_wandb_id(), type=str)
	parser.add_argument("--wandb-offline", action=BooleanOptionalAction, default=False, type=str)

	# Reproducibility
	parser.add_argument("--seed", action="store", type=int)

	args = parser.parse_args()

	if not args.log:
		args.wandb_mode = "disabled"

	if not args.model and not args.weights:
		raise RuntimeError("Cannot finetune a model without initial weights.")

	if args.seed is not None:
		seed_everything(args.seed)

	main(args) # TODO: SOLVE THIS (?) It ate all the RAM from arale
