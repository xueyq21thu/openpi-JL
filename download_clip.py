# download_clip.py
from transformers import CLIPProcessor, CLIPModel
import os

model_name = "openai/clip-vit-base-patch32"
save_directory = "./checkpoints/clip_model_local"
os.makedirs(save_directory, exist_ok=True)

print(f"Downloading model '{model_name}' to '{save_directory}'...")

# Processor 的下载通常不受影响
processor = CLIPProcessor.from_pretrained(model_name)
processor.save_pretrained(save_directory)

# --- THIS IS THE FIX ---
# Add use_safetensors=True to prioritize the secure format
# and bypass the torch version check.
model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
# --- END OF FIX ---

model.save_pretrained(save_directory)

print(f"✅ Download complete. The folder '{save_directory}' is ready.")