from smt_model.modeling_smt import SMTModelForCausalLM
from data import parse_kern_file

import numpy as np
import torch

from torchvision import transforms
from datasets import load_dataset

def convert_img_to_tensor(image):
    transform = transforms.Compose([
        transforms.RandomInvert(p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor()
	])

    image = transform(image)

    return image

def encode_transcription(transcription, w2i):
	print("Encoding transcription:")
	print(transcription)
	transcription = parse_kern_file(transcription, "bekern")
	print(transcription)
	return torch.tensor([w2i[t] for t in transcription], dtype=torch.long)

def decode_transcription(transcription, i2w):
	return torch.tensor([i2w[t] for t in transcription], dtype=torch.long)

def getDataset(datasetName: str, split: str):
	# Load dataset
	dataset = load_dataset("antoniorv6/"+datasetName, split=split)

	# Load vocabulary
	vocab_name: str = datasetName.replace("-", " ").title().replace(" ", "_")
	w2i = np.load(f"./vocab/{vocab_name}_BeKernw2i.npy", allow_pickle=True).item()
	i2w = np.load(f"./vocab/{vocab_name}_BeKerni2w.npy", allow_pickle=True).item()

	return dataset, w2i, i2w

def evaluateSMTPP(model, dataset, w2i, i2w, device = torch.device("cpu")):
	for sample in dataset:
		image = sample["image"]
		ground_truth = sample["transcription"]
		target = encode_transcription(ground_truth, w2i)

		predictions, _, logit_sequence = model.predict(convert_img_to_tensor(image).unsqueeze(0).to(device), convert_to_str=True)
		print("prediction:")
		print(predictions)
		print("ground_truth:")
		print(ground_truth)

def main():
	datasetName: str = "mozarteum"
	split: str = "train"
	dataset, w2i, i2w = getDataset(datasetName, split)

	print("w2i", type(w2i), ":", w2i)
	print("i2w", type(w2i), ":", i2w)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SMTModelForCausalLM.from_pretrained("antoniorv6/smtpp_mozarteum").to(device)

	evaluateSMTPP(model, dataset, w2i, i2w, device)

if __name__ == "__main__":
	main()
