import torch 
from diffusers import StableDiffusionPipeline

tokenizer = None 
device = None 

MY_TOKEN = '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'

MAX_NUM_WORDS = 77
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5


def init_model():
    global device
    global tokenizer

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
    ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=MY_TOKEN).to(device)
    tokenizer = ldm_stable.tokenizer

    return ldm_stable