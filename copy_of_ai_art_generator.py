from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

prompt = "vegetables"

if device == "cuda":
    with torch.autocast("cuda"):
        result = pipe(prompt)
else:
    result = pipe(prompt)

image = result.images[0]
image.save("ai_art.jpg")
print("✅ AI art saved as 'ai_art.jpg'")

image.show()

display(image)

image.save("art.jpg")
