import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image, make_image_grid
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16

# 原始图像加载（未修改）
original_image = load_image(
    "weight_prepare/humberger50-3-128/fp_reconstruction_15.png"
)

def process_image_to_tensor(image):
    # Convert PIL image to numpy array
    image = np.array(image)
    # Check if the image has only one channel (grayscale), and replicate channels to make it RGB
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack([image] * 3, axis=-1)  # Make it 3-channel by stacking
    elif image.shape[2] == 1:  # Single channel (e.g., depth map)
        image = np.concatenate([image, image, image], axis=2)  # Expand single channel to RGB
    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    # Convert numpy array to PyTorch tensor and permute to (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
    return image_tensor

original_image_tensor = process_image_to_tensor(original_image).unsqueeze(0).half().to(device)

# 修改部分：生成与原始图像尺寸相同的随机噪声
image_size = original_image.size  # 获取原始图像尺寸
noise_array = np.random.rand(image_size[1], image_size[0], 3) * 255  # 生成噪声（高度, 宽度, 通道）
image = torch.from_numpy(noise_array).permute(2, 0, 1).float() / 255.0  # 转换为张量格式并归一化
image = image.unsqueeze(0).to(device, dtype)

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

output = pipe(
    "A high-resolution, detailed version of a photograph of a hamberger on a wooden table with buildings in the background", image=image, control_image=original_image_tensor,
).images[0]

# 将 image 和 output 张量转换为 PIL 图像
image_pil = to_pil(original_image_tensor.squeeze(0).cpu())  # 转换为 PIL 图像
output_pil = to_pil(output)  # 生成的输出图像也转换为 PIL 图像
# 修改部分：保存生成的图像并显示输入图像为噪声
concat_image = make_image_grid([image_pil, output_pil], rows=1, cols=2)

# 保存图像
concat_image.save("output_image_with_noise.jpg")
