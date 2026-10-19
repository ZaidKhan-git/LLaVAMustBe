# pip install -U torchao  //if on google colab or kaggle
import json
import torch
import os
import torch.optim as optim
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler
from huggingface_hub import login, HfApi, hf_hub_download

"""
Downlaod Dataset:
.>pip install -U huggingface_hub
.>export HF_TOKEN=<your token>
.>hf download liuhaotian/LLaVA-CC3M-Pretrain-595K chat.json images.zip --repo-type dataset --set-your-dir data/cc3m
"""

# Directories -------------
cache_dir = "your-hugging-face-cache-dir"
data_dir = "set-path-to-cc3m-dataset" 
image_dir="set-path-to-extracted-image-dir"
local_dir = "set-dir-to-download-dataset-from-hugging-face"
DATASET_REPO = "liuhaotian/LLaVA-CC3M-Pretrain-595K" 
CHECKPOINT_DIR = "dir-to-save-checkpoints"

# ENV --------
USE_BF16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
USING_BLACKWELL = False # Enable if using Nvidia Blackwell GPU
SMOKE_TEST = False # Run before the final training strech
TEST_AFTER_TRAINING = True # Enable to run evaluation on projector after training

# Knobs ------
HUGGING_FACE_UPLOAD = False # Enable to upload checkpoints to your hugging face repo
REPO = "your-repo-path"
SPLIT_SIZE = 200000 # Slice large dataset (CC3M-595K). I was under strict hardware limitaions
PROJECTOR_DTYPE = torch.float32 # Change to torch.float16 if on Tensor GPU
# GLOBAL BATCH SIZE = NUM_EPOCHS * ACCUMULATION_STEP * BATCH_SIZE
NUM_EPOCHS=1 # Number of total epochs 
ACCUMULATION_STEP=8 # Mini Batch size
BATCH_SIZE=16 # Batch Size
MAX_LENGTH = 128 # MaX input tokens
SAVE_EVERY_STEPS = 512 # Saves checkpoints per SAVE_PER_STEP training steps
PRINT_EVERY_STEPS = 50 # Prints logs per PRINT_PER_STEP training steps
IMAGE_TOKEN = -200 # Used to mask IMAGE_TOKENS
# Autodetects needed dtype over different GPU architectures
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Using model precision: {MODEL_DTYPE}")

#-------------
# Pretrained Models ::::::::::::::::::::
LLM = "openai/clip-vit-large-patch14-336"
CLIP = "meta-llama/Llama-3.2-1B-Instruct"
#---------------------------------------

if USING_BLACKWELL:
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {torch.cuda.get_device_name(0)} | Compute Capability: {torch.cuda.get_device_capability(0)}")


os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
# Retrieve the token from Colab Secrets
hf_token = "your-hugging-face-token" 

# Import uploading and downloading client
hf_api = HfApi(token=hf_token)

# Log in to Hugging Face
login(token=hf_token)
print("Successfully Logged in.")

for fn in ["chat.json", "images.zip"]:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=fn,
        repo_type="dataset",
        local_dir=local_dir,                 # /workspace/LLaVA_CC3M_Data
        token=hf_token,   # use your new token, not the old one
    )
    print("Downloaded:", path)


image_processor = CLIPImageProcessor.from_pretrained(CLIP)
tokenizer = AutoTokenizer.from_pretrained(LLM)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class CC3MDataset(Dataset):
  def __init__(self, tokenizer, image_processor, max_length=MAX_LENGTH, data_dir=data_dir, image_dir=image_dir, split_size=SPLIT_SIZE):
    self.data_dir = data_dir
    self.image_dir = image_dir
    self._archive = None
    self.tokenizer = tokenizer
    self.image_processor = image_processor
    self.max_length = max_length
    self.bos_id = self.tokenizer.bos_token_id
    self.eos_id = self.tokenizer.eos_token_id
    self.pad_token_id = self.tokenizer.pad_token_id

    with open(os.path.join(data_dir, "chat.json"), "r", encoding='utf-8') as f:
      full_data = json.load(f)

    if split_size is not None:
      self.data = full_data[:split_size]
    else:
      self.data = full_data

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    image_filename = item["image"]
    try:
      with Image.open(os.path.join(self.image_dir, item["image"])) as im:
        raw_image = im.convert("RGB")
    except Exception:
        return self.__getitem__((idx + 1) % len(self.data))

    # Load conversations from item dict
    conversations=item["conversations"]

    # exclude captions from assistant to only calculate the loss ON
    captions = ""
    for turn in conversations:
      if turn["from"] != "human":
        captions = turn["value"].strip()
        break;

    # input_ids for captions (assistant answer)
    caption_ids = self.tokenizer.encode(
        captions,
        truncation=True,
        max_length=self.max_length - 3,
        add_special_tokens=False
    )

    # Build the required chat template format [BOS][IMG][CAPTION][EOT]
    seq_ids = [self.bos_id, IMAGE_TOKEN] + caption_ids + [self.eos_id]
    actual_len = len(seq_ids)
    pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_id

    # pad len to pad the ids till max_length
    pad_len = self.max_length - actual_len
    if pad_len >= 0:
      padded_seq_id = seq_ids + [pad_id] * pad_len

    # Stack the input_ids in tensors
    input_ids = torch.tensor(padded_seq_id, dtype=torch.long)

    # Build Labels from input_ids and mask out unrequired ids to disable loss calculation over them
    labels = input_ids.clone()
    labels[0] = -100 #mask BOS TOKEN
    labels[1] = -100 #mask IMAGE_TOKEN

    if pad_len > 0:
      labels[actual_len:] = -100

    attention_mask=torch.zeros(self.max_length, dtype=torch.long)
    # Only unmask assistant turn
    attention_mask[:actual_len] = 1

    image_input = self.image_processor(images=raw_image, return_tensors="pt")
    pixel_values = image_input.pixel_values.squeeze(dim=0) # Processed Image (CLIPImageProcessor)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
      }

def collate_fn(batch_list):
    return {
        key: torch.stack([s[key] for s in batch_list])
        for key in batch_list[0].keys()
    }

train_dataset = CC3MDataset(
    tokenizer,
    image_processor,
    max_length=MAX_LENGTH,
    data_dir=data_dir,
    image_dir=image_dir,
    split_size=SPLIT_SIZE
)

# Select number of worker processes depending upon cpu count
num_workers = min(8, os.cpu_count() or 4)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers, # Uses python multiprocessing
    persistent_workers=True, # Keep the zip local copy of each worker alive in between epochs
    pin_memory=True, # Speed up the tensor transfer from CPU RAM to GPU VRAM
    prefetch_factor=2,
    drop_last=True
)

print(f"Total Training Samples: {len(train_dataset)}")
print(f"Total batch per epoch: {len(train_loader)}")

class LLaVAModel(nn.Module):
  def __init__(self, vision_model_id=CLIP, llm_model_id=LLM, labels=None):
    super().__init__()

    # Load the clip vision model
    self.vision_encoder = CLIPVisionModel.from_pretrained(
        vision_model_id,
        torch_dtype=MODEL_DTYPE,
        attn_implementation="sdpa" # Uses flash attention behind the scenes
        )

    # Load the llama llm model
    self.llama_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        dtype = MODEL_DTYPE,
        attn_implementation="sdpa"  # Uses flash attention behind the scenes
    )
    # self.llama_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    self.llama_model.config.use_cache = False

  # Freeze CLIP and LLM 
    for params in self.vision_encoder.parameters():
      params.requires_grad = False

    for params in self.llama_model.parameters():
      params.requires_grad = False

    # Extract hidden sizes of CLIP vision model and LLM
    self.vision_hidden_shape = self.vision_encoder.config.hidden_size
    self.llama_hidden_shape = self.llama_model.config.hidden_size

    # Projction Layer to map raw outputs of CLIP Vision model to that of LLM. #Trainable
    self.projector = nn.Linear(in_features=self.vision_hidden_shape,
                               out_features=self.llama_hidden_shape)
    # Make sure the projector process data in native llama dtype of torch.bfloat16 or torch.float16
    self.projector.to(dtype=PROJECTOR_DTYPE) 

  def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):

    # Convert the pixel values and input ids into right shape and move to the current device
    pixel_values = pixel_values.to(dtype=MODEL_DTYPE, device=device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
      # Extract the output of the vision (CLIP) model over an sample image
      vision_output = self.vision_encoder(pixel_values=pixel_values, output_hidden_states=True)
      # Extract the right shape of the output of vision_model
      image_features = vision_output.hidden_states[-2][:, 1:, :]

    # Project the raw image output into LLM hyper space
    projected_image_features = self.projector(image_features).to(dtype=MODEL_DTYPE)

    batch_size = input_ids.shape[0]
    num_image_tokens = projected_image_features.shape[1]

    # Seprately account for bos ids and caption ids to build LLAMA style template
    bos_id = input_ids[:, :1]
    caption_ids = input_ids[:, 2:]

    # Get embeddings from the text model
    embed_tokens = self.llama_model.model.embed_tokens
    bos_embed = embed_tokens(bos_id)
    caption_embed = embed_tokens(caption_ids)

    # Concatente to get final formatted embeddings
    multimodal_embeddings = torch.cat([bos_embed, projected_image_features, caption_embed], dim=1)

    # Seprately account for bos token to build LLAMA style template
    bos_mask = attention_mask[:, :1]
    # Seprately account for iamge tokens to build LLAMA style template
    image_attention = torch.ones((batch_size, num_image_tokens), dtype=attention_mask.dtype, device=input_ids.device)
    # Seprately account for caption tokens to build LLAMA style template
    caption_mask = attention_mask[:, 2:]
    multimodal_attention_mask = torch.cat([bos_mask, image_attention, caption_mask], dim=1)

    bos_labels = torch.full((batch_size, 1), fill_value=-100, dtype=labels.dtype, device=input_ids.device)
    image_labels = torch.full((batch_size, num_image_tokens), fill_value = -100, dtype=labels.dtype, device=input_ids.device)
    caption_labels = labels[:, 2:]
    multimodal_labels = torch.cat([bos_labels, image_labels, caption_labels], dim=1)

    # Pass the embeddings to the llama text model
    outputs = self.llama_model(
        inputs_embeds=multimodal_embeddings,
        attention_mask=multimodal_attention_mask,
        labels=multimodal_labels
    )
    return outputs

# Create the instance of the model
model = LLaVAModel()
model.vision_encoder.eval()
model.llama_model.train()

model = model.to(device)
model.projector.train()

optimizer = optim.AdamW(model.projector.parameters(), lr=1e-3, weight_decay=0.0, fused=True)

# calcuations ---
total_optimizer_steps = (len(train_loader) // ACCUMULATION_STEP)
warmup_steps = int(0.03 * total_optimizer_steps)

# Enables Dynamic Learning Rate -> Visual Instruction Tuning (2023)
lr_schedular = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps
)

# Used to prevent gradient underflow during Mixed Precision Training
scaler = torch.amp.GradScaler("cuda", enabled=(MODEL_DTYPE == torch.float16))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "projector_weights_")

# Saves the last version of model and training variables at certain step where it is called
def save_checkpoint(model, optimizer, lr_schedular, scaler, epoch, step, loss, filepath=CHECKPOINT_PATH):
  checkpoints = {
      "epoch":epoch,
      "projector_state_dict":model.projector.state_dict(),
      "optimizer_state_dict":optimizer.state_dict(),
      "schedular_state_dict":lr_schedular.state_dict(),
      "scaler_state_dict": scaler.state_dict(),
      "step":step,
      "loss":loss,
  }
  torch.save(checkpoints, filepath)
  print(f"Checkpoint saved at Epoch: {epoch+1} and Step: {step+1} at {filepath}")

start_epoch = 0
start_step = 0

# Restoring last checkpoint if training is intruppted
if os.path.exists(CHECKPOINT_PATH):
  print(f"Checkpoint found at {CHECKPOINT_PATH}")
  checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

  model.projector.load_state_dict(checkpoint["projector_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  lr_schedular.load_state_dict(checkpoint["schedular_state_dict"])
  scaler.load_state_dict(checkpoint["scaler_state_dict"])

  start_epoch = checkpoint["epoch"]
  start_step = checkpoint["step"] + 1
  loss = checkpoint["loss"]
  print("SUCCESSFULLY RESTORED THE LAST CHECKPOINT")
else:
  print("!NO checkpoint found. Initializing fresh training...")

# Manually setting the optimizer internal state to the target device - GPU
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

# Cell A: missing-image check #claude
missing = [it["image"] for it in tqdm(train_dataset.data)
           if not os.path.exists(os.path.join(train_dataset.image_dir, it["image"]))]
print(f"Missing: {len(missing)} / {len(train_dataset.data)}", missing[:5])

# Cell B: collate_fn sanity check #claude
b = collate_fn([train_dataset[i] for i in range(4)])
for k, v in b.items(): print(k, tuple(v.shape), v.dtype)
assert b["pixel_values"].shape == (4, 3, 336, 336)
assert (b["input_ids"][:, 1] == IMAGE_TOKEN).all()
assert torch.isfinite(b["pixel_values"]).all()
print(tokenizer.decode(b["input_ids"][0][2:][b["attention_mask"][0][2:].bool()]))

for epoch in range(start_epoch, NUM_EPOCHS):
  running_loss = 0.0
  progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

  # Clear all the old gradients. Torch does not automatically does this. 
  optimizer.zero_grad()
  for step, batch in progress_bar:
    if epoch == start_epoch and step < start_step:
      continue

    with torch.amp.autocast(device, dtype=MODEL_DTYPE):
      # Send all the inputs to the respective device
      pixel_values = batch["pixel_values"].to(device, non_blocking=True)
      input_ids = batch["input_ids"].to(device, non_blocking=True)
      attention_mask = batch["attention_mask"].to(device, non_blocking=True)
      labels = batch["labels"].to(device,  non_blocking=True)

      output = model(
          pixel_values=pixel_values,
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )

      # Calculate the loss (Cross Entropy)
      loss = output.loss
      scaled_loss = loss / ACCUMULATION_STEP
      if not torch.isfinite(loss):
        print(f"inf loss found on Step {step + 1}, skipping..")
        optimizer.zero_grad()
        continue

    # Backpropogation
    scaler.scale(scaled_loss).backward()
    running_loss += scaled_loss.item()

    # Clip the gradients below 1 to prevent exploding gradient problem
    if (step+1) % ACCUMULATION_STEP == 0:
      scaler.unscale_(optimizer)
      gn = torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
      if torch.isfinite(gn):
        # Ajust the parameters via optimizer
        scaler.step(optimizer)
      else:
        print("Skipping, inf found in weights")
      scaler.update()
      lr_schedular.step()
      # Clean up old gradients
      optimizer.zero_grad()

    progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}",
                              "Current Learning Rate": f"{lr_schedular.get_last_lr()[0]:.6f}"})

    if (step+1) % PRINT_EVERY_STEPS == 0 and step > 0:
            avg_loss = (running_loss / PRINT_EVERY_STEPS) * ACCUMULATION_STEP
            print(f" Step {step+1} | Rolling Avg Alignment Loss: {avg_loss:.4f} | Current_Learning_Rate: {lr_schedular.get_last_lr()[0]:.6f} | gn: {gn.item():.3f}")
            running_loss = 0.0
#------------------------------------------------------------------------------------------------------------------------------------------------------
    if SMOKE_TEST and (step+1) % ACCUMULATION_STEP == 0:
      print(f"step {step+1} | loss {loss.item():.3f} | gn {gn.item():.3f} | peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    if SMOKE_TEST and step >= 23:
      break
#------------------------------------------------------------------------------------------------------------------------------------------------------
    if (step+1) % SAVE_EVERY_STEPS == 0:
      save_checkpoint(model, optimizer, lr_schedular, scaler, epoch, step, loss.item(), filepath=CHECKPOINT_PATH)
      if HUGGING_FACE_UPLOAD:
        try:
          hf_api.upload_file(
              path_or_fileobj=CHECKPOINT_PATH,
              path_in_repo="Projector_Training_Checkpoint",
              repo_id=REPO,
              repo_type="dataset",
              commit_message=f"Checkpoint Saved at Step: {step+1}"
          )
        except Exception as e:
          print("!Upload to Hugging Face Hub failed. ", e)

if not SMOKE_TEST:
  projector_save_path = os.path.join(CHECKPOINT_DIR, "projector.pt")
  torch.save(model.projector.state_dict(), projector_save_path)
  if HUGGING_FACE_UPLOAD:
    try:
      hf_api.upload_file(
                path_or_fileobj=projector_save_path,
                path_in_repo="Trained Projector",
                repo_id=REPO,
                repo_type="dataset",
                commit_message=f"Checkpoint Saved at Step: {step+1}"
            )
      print(f"\n ):-> Feature Alignment complete! Projector weights saved directly to: {projector_save_path}")
    except Exception as e:
      print("!Upload to Hugging Face Hub failed. ", e)

#this block is build with claude
if TEST_AFTER_TRAINING:
  import random, numpy as np
  from IPython.display import display

  model.eval()
  model.llama_model.config.use_cache = True
  DATA_DIR = data_dir
  IMAGE_DIR = image_dir
  with open(os.path.join(DATA_DIR, "chat.json"), encoding="utf-8") as f:
      full_data = json.load(f)
  held_out = full_data[200000:]          

  # This function captions the randomly new image from the dataset. Try for 5-6 samples and verify.
  @torch.no_grad()
  def caption_image(image, max_new_tokens=40):
      pv = image_processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
      pv = pv.to(device, dtype=MODEL_DTYPE)
      feats = model.vision_encoder(pixel_values=pv, output_hidden_states=True).hidden_states[-2][:, 1:, :]
      img_emb = model.projector(feats.float()).to(MODEL_DTYPE)
      bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
      bos_emb = model.llama_model.get_input_embeddings()(bos)
      embeds = torch.cat([bos_emb, img_emb], dim=1)
      attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
      out = model.llama_model.generate(
          inputs_embeds=embeds, attention_mask=attn,
          max_new_tokens=max_new_tokens, do_sample=False,
          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
      )
      return tokenizer.decode(out[0], skip_special_tokens=True)

  # Select 8 images from unseen part of the dataset
  random.seed(0)
  for it in random.sample(held_out, 8):
      img = Image.open(os.path.join(IMAGE_DIR, it["image"]))
      gt = next(t["value"] for t in it["conversations"] if t["from"] != "human")
      display(img.resize((256, 256)))
      print("GT  :", gt.strip())
      print("PRED:", caption_image(img), "\n")

  # Check on pure black images
  # Controls: captions of content-free images
  print("BLACK:", caption_image(Image.new("RGB", (336, 336), (0, 0, 0))))
  noise = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
  print("NOISE:", caption_image(noise))

