# import libraries

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

# load model

processor =BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# load test images
image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# image to tensor
inputs = processor(image,return_tensors="pt")


# create a caption text and model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    output = model.generate(**inputs)


# token to human readable caption
caption = processor.decode(output[0], skip_special_tokens=True)
print(caption)
