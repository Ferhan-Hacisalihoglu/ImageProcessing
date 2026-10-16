# import libraries
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests
import torch

# model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


# VIT Processor
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


# AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


# get visual url and request
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# image to tensor
pixel_values = processor(image, return_tensors="pt").pixel_values

# model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
pixel_values = pixel_values.to(device)

# run model 
outputs = model.generate(pixel_values, max_length=64)

# print results
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(caption)