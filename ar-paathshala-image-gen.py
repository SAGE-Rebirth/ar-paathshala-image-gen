from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

def ArPaathshsalaImageApp(prompt):
  model_id = "prompthero/openjourney"
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  pipe = pipe.to("cuda")

  prompt = prompt
  image = pipe(prompt).images[0]
  return image


app = gr.Interface(ArPaathshsalaImageApp,
                   inputs = "text",
                   outputs = "image",
                   theme = gr.themes.Soft(),
                   title = "AR Paathshala Image Generator")

app.launch()