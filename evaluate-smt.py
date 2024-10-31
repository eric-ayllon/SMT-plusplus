from typing import Union, Sequence, Optional, Dict, Tuple, List
from torch import Tensor

from smt_model.modeling_smt import SMTModelForCausalLM
from smt_trainer import SMTPP_Trainer
from data import parse_kern_file
from torch.cuda import is_available as cuda_enabled

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from torch import\
                int as torchint,\
                zeros

from wandb.sdk.wandb_run import Run
from wandb.sdk.lib import RunDisabled

import numpy as np
import pandas as pd
import torch
import wandb

from torchvision import transforms
from datasets import load_dataset, concatenate_datasets

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

# def getDataset(datasetName: str, split: str):
def getDataset(datasetName: str):
    # Load dataset
    ds = load_dataset("antoniorv6/"+datasetName)
    # dataset = pd.concat([ds["train"], ds["val"], ds["test"]])
    dataset = concatenate_datasets([ds["train"], ds["val"], ds["test"]])

    # Load vocabulary
    vocab_name: str = datasetName.replace("-", " ").title().replace(" ", "_")
    w2i = np.load(f"./vocab/{vocab_name}_BeKernw2i.npy", allow_pickle=True).item()
    i2w = np.load(f"./vocab/{vocab_name}_BeKerni2w.npy", allow_pickle=True).item()

    return dataset, w2i, i2w

def getLogger(args: Namespace) -> Union[Run, RunDisabled]:
    # Prepare the logger
    logger = wandb.init(
                                        entity="eap-team",
                                        project=args.project,
                                        id=args.run_id,

                                        name=args.run_name,
                                        notes="Data for the prediction confidence evaluation.",

                                        config={**vars(args)},
                                        mode=args.run_mode
                                        )

    if args.run_mode != "disabled":
        print(f"WandB ID: {args.run_id}; Mode: {args.run_mode}")

    return logger

def evaluateSMTPP(model, dataset, w2i, i2w, logger: Union[Run, RunDisabled], device = torch.device("cpu")):
    model.eval()

    logger.define_metric("total edit distance")
    logger.define_metric("total length")
    logger.define_metric("total SER")

    # logger.define_metric("logits", step_metric="sample") # logits
    logger.define_metric("confidences", step_metric="sample") # framewise confidences
    logger.define_metric("prediction", step_metric="sample") # decoded prediction
    logger.define_metric("target", step_metric="sample") # target
    logger.define_metric("edit distance", step_metric="sample") # sample edit distance

    totalDistance: int = 0
    totalLength: int = 0
    totalSER: float = .0

    for sample_idx, sample in enumerate(dataset):
        image = sample["image"]
        # print("Sample size:", image)
        ground_truth = sample["transcription"]
        target = tokenize_transcription(ground_truth)

        # print(torch.cuda.mem_get_info(device))
        image_tensor = convert_img_to_tensor(image).unsqueeze(0).to(device)
        # print(torch.cuda.mem_get_info(device))

        # predictions, _, logit_sequence = model.predict(image_tensor, convert_to_str=True)
        predictions, _, logit_sequence = model.predict(image_tensor)

        # print("prediction:")
        # print(predictions)
        # print("target:")
        # print(target)
        # print("ground_truth:")
        # print(ground_truth)

        # Get confidences
        # logits (1, length, vocabulary) -> probabilities (1, length, vocabulary) -> confidences (1, length)
        confidences = logit_sequence.softmax(dim=-1).max(dim=-1)[0].tolist()

        distance = levenshtein(predictions, target)[0]
        totalDistance += distance
        totalLength += len(target)

        logger.log({
                    "sample": sample_idx,
                    # # "logits": logits[sample_idx, :widths[sample_idx]].tolist(),
                    "confidences": confidences,
                    "prediction": predictions,
                    "target": target,
                    "edit distance": distance,
                    "length": len(target),
                    # # "SER": 100.0*distance/length,
                    # "decoded frames": decoded_frames.tolist(),
                    # "edit operations": operations.tolist(),
                    # "frame errors": frame_errors.tolist(),
                    })

def main(args: Namespace):
    logger = getLogger(args)

    # for split in args.dataset_splits:
        # dataset, w2i, i2w = getDataset(args.dataset_name, split)

        # print("w2i", type(w2i), ":", w2i)
        # print("i2w", type(w2i), ":", i2w)

        # # model = SMTModelForCausalLM.from_pretrained("antoniorv6/smtpp_mozarteum").to(args.device)
        # # model = SMTPP_Trainer(maxh=2512, maxw=2512, maxlen=5512, out_categories=191, padding_token=0, in_channels=1, w2i=w2i, i2w=i2w, d_model=256, dim_ff=256, num_dec_layers=8).load_from_checkpoint("./weights/SMTPP_Mozarteum_Synthetic.ckpt").model.to(args.device)
        # model = SMTPP_Trainer.load_from_checkpoint("./weights/SMTPP_Mozarteum_Synthetic.ckpt").model.to(args.device)

        # evaluateSMTPP(model, dataset, w2i, i2w, logger, args.device)
    dataset, w2i, i2w = getDataset(args.dataset_name)

    # print("w2i", type(w2i), ":", w2i)
    # print("i2w", type(w2i), ":", i2w)

    model = SMTPP_Trainer.load_from_checkpoint("./weights/SMTPP_Mozarteum_Synthetic.ckpt").model.to(args.device)

    evaluateSMTPP(model, dataset, w2i, i2w, logger, args.device)

if __name__ == "__main__":
    parser = ArgumentParser(
                                            prog="OMR SMTPP",
                                            description="Python script for evaluating the SMTPP.",
                                            epilog=""
                                            )

    # Data
    parser.add_argument("--dataset-name", action="store", default="mozarteum", type=str) # Default to mozarteum
    parser.add_argument("--dataset-splits", action="store", default=["train", "val", "test"], type=str, nargs="+") # Default to only all

    # Model
    parser.add_argument("--model", action="store", default="antoniorv6/smtpp_mozarteum", type=str)
    parser.add_argument("--device", action="store", default="cuda" if cuda_enabled() else "cpu", type=str)

    # Logging
    parser.add_argument('--log', action=BooleanOptionalAction, default=True, type=bool)
    parser.add_argument("--project", action="store", default="Test", type=str)
    parser.add_argument("--run-name", action="store", type=str)
    parser.add_argument("--run-id", action="store", default=wandb.util.generate_id(), type=str)
    parser.add_argument("--run-mode", action="store", default="online", type=str)

    args = parser.parse_args()

    if not args.log:
        args.run_mode = "disabled"

    main(args)
