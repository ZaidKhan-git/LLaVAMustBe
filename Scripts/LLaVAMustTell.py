# pip install -U torchao  //if on google colab or kaggle
import torch
import os
from PIL import Image
from torch import nn
from safetensors.torch import load_file
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from huggingface_hub import hf_hub_download, snapshot_download, login, HfApi
from IPython.display import display

# Directories ---------------------------
cache_dir = "/workspace/huggingface_cache"
lora_path = "/content/Final_LoRA Adapter Weights/Final_LoRA Adapter Weights/adapter_config.json"
projector_path = "/content/LLaVA Final Projector Weights.pt"

# Select From Weights--
projector_filename = "your-projector-weights-filename-name-on-huggingface-dataset"
lora_folder_path = "your-lora-adapter-folder-name-on-huggingface-dataset"
DATASET_REPO = "KhanZaid-NL/LLaVA_Weights"   # Change if using your own weights

# Set It Yourself-------------
USE_PRETRAINED_WEIGHTS = True
REPO = "KhanZaid-NL/LLaVA_Weights"
USE_BF16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
# Automatically detect dtype according to GPU Architectur
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Using model precision: {MODEL_DTYPE}")

# Pretrained Model ----------------------
CLIP = "openai/clip-vit-large-patch14-336"
LLM = "meta-llama/Llama-3.2-1B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Hugging Face Setup----------------
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
hf_token = "your_hugging_face_token"
# Log in to Hugging Face
login(token=hf_token)
hf_api = HfApi(token=hf_token)
print("Successfully Logged in")

if USE_PRETRAINED_WEIGHTS:
  projector_filename = "LLaVA Final Projector Weights.pt"
  lora_folder_path = "Final_LoRA Adapter Weights/*"
path1 = hf_hub_download(
    repo_id=DATASET_REPO,
    filename=projector_filename,
    repo_type="dataset",
    local_dir="/content",
    token=os.environ.get("hf_token") 
)
print("Projector Downloaded:", path1)

path2 = snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    allow_patterns=lora_folder_path,  # Only download this folder
    local_dir="./Final_LoRA Adapter Weights"
)
print("LoRA Weights Downloaded", path2)

# Define Image Processor and Tokenzier
image_processor = CLIPImageProcessor.from_pretrained(CLIP)
tokenizer = AutoTokenizer.from_pretrained(LLM)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

class LLaVAStageInference(nn.Module):
  def __init__(self,
                lora_path = lora_path,
                llm_model_id=LLM,
                clip_vision_model_id=CLIP):
    super().__init__()

    # Load the vision model in bfloat16 precision
    self.clip_vision_model = CLIPVisionModel.from_pretrained(
      clip_vision_model_id,
      torch_dtype=MODEL_DTYPE
    )
    for params in self.clip_vision_model.parameters():
      params.requires_grad = False

    # Base LLaMA in native bfloat16
    base_llama = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype=MODEL_DTYPE,
        attn_implementation="sdpa"
    )
    base_llama.config.use_cache = False

    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
          "q_proj", "k_proj", "v_proj", "o_proj",
          "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
      )
    self.llama_model = get_peft_model(base_llama, lora_config)

    # Projector matching bfloat16
    hidden_clip = self.clip_vision_model.config.hidden_size
    hidden_llama = self.llama_model.config.hidden_size

    self.projector = nn.Linear(hidden_clip, hidden_llama).to(device, dtype=MODEL_DTYPE)
    for params in self.projector.parameters():
      params.requires_grad = False

  def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
    device = self.llama_model.device
    input_ids = input_ids.to(device)

    # 1. Vision extraction in MODEL_DTYPE
    with torch.no_grad():
      pixel_values = pixel_values.to(device, dtype=MODEL_DTYPE)
      vision_output = self.clip_vision_model(pixel_values=pixel_values, output_hidden_states=True)
      image_features = vision_output.hidden_states[-2][:, 1:, :]

    # Text embedding lookup in bfloat16
    embed = self.llama_model.get_input_embeddings()
    bos_embed = embed(input_ids[:, :1])
    text_embed = embed(input_ids[:, 1:])
    img_embed = self.projector(image_features.float()).to(bos_embed.dtype)
    input_embeds = torch.cat([bos_embed, img_embed, text_embed], dim=1)

    batch_size = input_ids.shape[0]
    num_image_tokens = img_embed.shape[1]

    # Combine the attention_mask and labels
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        image_attention = torch.ones(
            (batch_size, num_image_tokens),
            dtype=attention_mask.dtype,
            device=device
        )
        attention_mask = torch.cat([attention_mask[:, :1], image_attention, attention_mask[:, 1:]], dim=1)

    if labels is not None:
      labels = labels.to(device)
      image_labels = torch.full(
            (batch_size, num_image_tokens),
            fill_value=-100,
            dtype=labels.dtype,
            device=device
        )
      labels = torch.cat([labels[:, :1], image_labels, labels[:, 1:]], dim=1)

    output = self.llama_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        labels=labels
    )
    return output

# Inintiate the Model
model = LLaVAStageInference()

# Load the Weights ->
if os.path.exists(projector_path) and os.path.exists(lora_path):
  res = set_peft_model_state_dict(model.llama_model, load_file(lora_path))
  print("LoRA Loaded.")
  checkpoint = torch.load(projector_path, map_location=device, weights_only=False)
  state = checkpoint() if callable(checkpoint) else checkpoint
  model.projector.load_state_dict(state)
  print("Projector Loaded")

# Define Chatting Logic
history = "" # Context : current architecture does not account for context yet.
def chat(image, questions, max_tokens=512):
  model.to(device).eval()
  model.projector = model.projector.to(dtype=torch.float16)
  px = image_processor(images=image, return_tensors="pt").pixel_values.to(device=device, dtype=MODEL_DTYPE)
  feat = model.clip_vision_model(pixel_values=px, output_hidden_states=True).hidden_states[-2][:, 1:, :]
  img = model.projector(feat.to(dtype = model.projector.weight.dtype, device=device))
  embd = model.llama_model.get_input_embeddings()
  bos = embd(torch.tensor([[tokenizer.bos_token_id]], device=device))
  eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
  pad = tokenizer.convert_tokens_to_ids("<|eot_id|>")


  history = ""
  for ques in questions:
    history += ques
    clean_txt = f"<|start_header_id|>user<|end_header_id|>\n\n{ques}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inp_ids = tokenizer(clean_txt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    embeds = torch.cat([bos, img, embd(inp_ids)], dim=1).to(device)
    output = model.llama_model.generate(
        inputs_embeds=embeds,
        attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device),
        max_new_tokens=max_tokens, pad_token_id = pad, do_sample=False, use_cache=True
    )
    ans = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    history += f"{ans}<|eot|>"

    return ans

imge = Image.open("/content/coco.jpg")
display(imge.resize((256, 256)))

print("Instructions: (new) to reset chat, (exit) to exit ")
while True:
  prompt = input("Enter Prompt: ")
  if prompt == "exit": print("\nExited. Context is saved \n"); break;
  if prompt == "new": print("\nNew Chat. Context rests. \n"); history = ""; continue
  print("--------------------------------------------------------------------------------------------")
  print(f"\nAnswer: {chat(imge, [prompt])}", "\n")
  print("--------------------------------------------------------------------------------------------")

