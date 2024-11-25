import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# 设备设置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 1. 加载 VAE（变分自编码器）
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)

# 2. 加载 UNet（U-Net 模型）
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)

# 3. 加载 DDIM 调度器
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# 4. 加载 CLIP 模型和预处理器
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

# 5. 加载本地图像作为图像条件
image_path = "weight_prepare/humberger64-3-128/fp_reconstruction_15.png"
image_prompt = Image.open(image_path).convert("RGB")

# 6. 文本条件
text_prompt = "a photograph of a hamburger in the center of a wooden table with high buildings in the background"
negative_prompt = "blurry, low quality, bad composition"  # 消极提示词

# 7. 预处理输入数据（文本）
text_input = clip_tokenizer([text_prompt], padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt")
negative_input = clip_tokenizer([negative_prompt], padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt")

# 8. 提取文本特征
with torch.no_grad():
    text_embeddings = clip_text_model(text_input.input_ids.to(device))[0].to(device)
    negative_embeddings = clip_text_model(negative_input.input_ids.to(device))[0].to(device)

# 9. 标准化特征
text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
negative_embeddings = negative_embeddings / negative_embeddings.norm(p=2, dim=-1, keepdim=True)

# 10. 加载图像条件
image_input = clip_processor(images=image_prompt, return_tensors="pt", padding=True)
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_input['pixel_values'].to(device))

# 11. 标准化图像特征
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# 12. 准备初始噪声图像
latent_image = torch.randn((1, 4, 64, 64)).to(device)

# 13. 扩散过程：从噪声图像开始并迭代生成图像
num_inference_steps = 50
guidance_scale = 7.5  # 文本条件强度
strength = 0.75  # 图像条件强度

# 使用 DDIM 调度器初始化扩散进程
scheduler.set_timesteps(num_inference_steps)

# 14. 迭代生成过程
for t in scheduler.timesteps:
    latent_image = latent_image.detach()  # 解除梯度
    noisy_latents = scheduler.add_noise(latent_image, torch.randn_like(latent_image).to(device), t)
    
    # 拼接文本和图像特征作为条件（合并正向和负向提示词）
    image_features = image_features.unsqueeze(1).repeat(1, text_embeddings.size(1), 1) # 使用文本的(batch_size, sequence_length, embedding_dim)，中sequence_length的大小扩展图像特征

    # 拼接特征
    input_cond = torch.cat([text_embeddings, image_features, negative_embeddings], dim=0)
    input_text_cond = torch.cat([text_embeddings, negative_embeddings], dim=0)
    
    # UNet 进行噪声预测
    noise_pred = unet(noisy_latents, t, encoder_hidden_states=input_text_cond).sample

    # 更新潜在图像
    latent_image = scheduler.step(noise_pred, t, noisy_latents).prev_sample

# 15. 解码图像
generated_image = vae.decode(latent_image).sample

# 16. 将生成的图像转换为可显示的格式
generated_image = generated_image.clamp(0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
generated_image = (generated_image * 255).astype("uint8")
generated_image_pil = Image.fromarray(generated_image)

# 展示生成的图像
generated_image_pil.show()
