from transformers import CLIPProcessor, CLIPModel

# model name: # https://huggingface.co/openai/clip-vit-base-patch32
model_name = "openai/clip-vit-base-patch32"

save_directory = "./checkpoints/clip_model_local" 

print(f"Downloading model '{model_name}' to '{save_directory}'...")

# 下载并保存 Processor 和 Model 的所有文件
processor = CLIPProcessor.from_pretrained(model_name)
processor.save_pretrained(save_directory)

model = CLIPModel.from_pretrained(model_name)
model.save_pretrained(save_directory)

print("✅ Download complete.")