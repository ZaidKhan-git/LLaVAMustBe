# pip install -U torchao  //if on google colab or kaggle
import os
import json
import random
import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torch.amp import autocast, GradScaler
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from huggingface_hub import login
from huggingface_hub import HfApi

"""
Downlaod Dataset:
.>pip install -U huggingface_hub
.>export HF_TOKEN=<your token>
.>hf download liuhaotian/LLaVA-Instruct-150K conversation_58k.json detail_23k.json complex_reasoning_77k.json --repo-type dataset --set-your-dir data/instruct
.>wget http://images.cocodataset.org/zips/train2017.zip
.>wget http://images.cocodataset.org/zips/val2017.zip
"""

# Directories------
SAVE_DIR = "set-your-local-saving-directory" 
cache_dir = "set-your-hugging-face-directory"
conv_dir = "set-your-conversation-data-path"
detail_dir = "set-your-detail-data-path"
complex_dir = "set-your-complex-data-path"
image_dir = "set-your-coco-images-path"
projector_dir = "set-the-path-to-load-projector-from-LLaVAMustBeCold"
train_state_path = "set-path-to-save-checkpoints-during-training"
sd = "set-path-to-load-trained-lora-adapter(if-any)"
val_dir = "set-your-coco-images-validation-path"

#----------
# ENV------
SMOKE_TEST = False # Run before the final training strech
TRAINING_MODE = False # Enable if training
ENABLE_TEST_DURING_TRAINING = False # Enable if tests to check if pipeline is working as intented
USING_BLACWELL_GPU = True # If using mordern GPU (Nvidia Blackwell)
HUGGING_FACE_UPLOAD = True # Enables to upload weights to your hugging face repo
MY_REPO = "your-hugging-face-repository" # You hugging face repo
#------------------
# KNOBS -----------
NUM_EPOCHS = 1   # Set desired number of epochs
BATCH_SIZE = 4   # Set batch size to load dataloader
ACCUMULATION_STEP = 8 # Sets Mini Batch Size
SAVE_PER_STEP = 512 # Saves checkpoints per SAVE_PER_STEP training steps
PRINT_PER_STEP = 64 # Prints logs per PRINT_PER_STEP training steps
MAX_LENGTH = 1024 # Maximum token length needed (depends on GPU VRam)
# Autodetects needed dtype over different GPU architecture
USE_BF16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"Using model precision: {MODEL_DTYPE}")

#------------ Select Pre Trained Models ------------
LLM = "meta-llama/Llama-3.2-1B-Instruct" # I used this due to my hardware limitations, you may use Vicuna 7B or 13B
CLIP = "openai/clip-vit-large-patch14-336"
#-------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if TRAINING_MODE:
    if USING_BLACWELL_GPU:
        # Enable Blackwell TF32 & CUDA Allocator optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

    # Retrieve the token from Colab Secrets
    hf_token = "your-hugging-face-token"
    # Log in to Hugging Face
    login(token=hf_token)

    hf_api = HfApi(token=hf_token)

    print("Successfully Logged in")

    with open(conv_dir, "r") as f:
        conv_data = json.load(f)
    with open(detail_dir, "r") as f:
        detail_data = json.load(f)
    with open(complex_dir, "r") as f:
        complex_data = json.load(f)

    CONVERSATION_DATA_LENGTH = len(conv_data)
    DETAILED_DESCRIPTION_DATA_LENGTH = len(detail_data)
    COMPLEX_REASONING_DATA_LENGTH = len(complex_data)
    print("Loading and filtering JSON files")
    print(f"Selected Samples from Conversations: {CONVERSATION_DATA_LENGTH}")
    print(f"Selected Samples from Detailed Descriptions: {DETAILED_DESCRIPTION_DATA_LENGTH}")
    print(f"Selected Samples from Complex Reasoning: {COMPLEX_REASONING_DATA_LENGTH}")

    random.seed(42)
    custom_dataset = (
        random.sample(conv_data, CONVERSATION_DATA_LENGTH) +
        random.sample(detail_data, DETAILED_DESCRIPTION_DATA_LENGTH) +
        random.sample(complex_data, COMPLEX_REASONING_DATA_LENGTH)
    )
    print("Successfully filtered the Dataset")

class Custom50KMultiTurnDataset(Dataset):
  def __init__(self, data_list, image_dir=image_dir):
    self.image_dir=image_dir
    self.data=data_list

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx].copy()

    # Load image safely without tokenizing or transforming
    image_filename = os.path.basename(item["image"])
    image_path = os.path.join(self.image_dir, image_filename)
    try:
        with Image.open(image_path) as img:
            raw_image = img.convert("RGB")
    except Exception:
        raw_image = self.__getitem__((idx + 1) % len(self.data))

    return {
        "image": raw_image,
        "conversations": item["conversations"]
        }

class MultiTurnCollater:
    def __init__(self, image_processor, tokenizer, max_length=1024):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_header_ids = self.tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens = False).input_ids
        self.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch_list):
        raw_images = [s['image'] for s in batch_list]
        pixel_values = self.image_processor(
            images=raw_images,
            return_tensors="pt"
        ).pixel_values

        batch_input_ids = []
        batch_labels = []

        # Process conversation sequences
        for sample in batch_list:
            all_input_ids = [self.bos_token_id]
            all_labels_ids = [-100]

            for turn in sample["conversations"]:
                role = turn["from"]
                content = turn["value"]

                if role == "human":
                    clean_text = content.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()
                    prompt_str = f"<|start_header_id|>user<|end_header_id|>\n\n{clean_text}<|eot_id|>"
                    prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids

                    all_input_ids.extend(prompt_ids)
                    all_labels_ids.extend([-100] * len(prompt_ids))

                elif role == "gpt":
                    all_input_ids.extend(self.assistant_header_ids)
                    all_labels_ids.extend([-100] * len(self.assistant_header_ids))

                    response_str = f"{content}<|eot_id|>"
                    response_ids = self.tokenizer(response_str, add_special_tokens=False).input_ids

                    all_input_ids.extend(response_ids)
                    all_labels_ids.extend(response_ids)

            # Truncate
            all_input_ids = all_input_ids[:self.max_length]
            all_labels_ids = all_labels_ids[:self.max_length]

            batch_input_ids.append(all_input_ids)
            batch_labels.append(all_labels_ids)

        batch_max_tokens = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_labels = []

        for id, lbs in zip(batch_input_ids, batch_labels):
            pad_len = batch_max_tokens - len(id)
            padded_input_ids.append(id + [self.pad_token_id] * pad_len)
            padded_labels.append(lbs + [-100] * pad_len)

        # Stack into batch tensors
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        labels = torch.tensor(padded_labels, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_input_ids), batch_max_tokens, dtype=torch.long)
        for i, ids in enumerate(batch_input_ids):
          attention_mask[i, :len(ids)] = 1

        return {

            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class LLaVAStage2Model(nn.Module):
    def __init__(self, projector_path=projector_dir if TRAINING_MODE else None,
                 llm_model_id=LLM,
                 clip_vision_model_id=CLIP):
        super().__init__()

        # Load the vision model in bfloat16 precision
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            clip_vision_model_id,
            torch_dtype=MODEL_DTYPE,
        )
        for params in self.clip_vision_model.parameters():
            params.requires_grad = False

        # Base LLaMA in native bfloat16
        base_llama = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=MODEL_DTYPE,
            attn_implementation="sdpa",
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
        self.llama_model.print_trainable_parameters()

        # 4. Projector matching bfloat16
        hidden_clip = self.clip_vision_model.config.hidden_size
        hidden_llama = self.llama_model.config.hidden_size

        self.projector = nn.Linear(hidden_clip, hidden_llama).to(device, dtype=torch.float32)
        if TRAINING_MODE:
            print(f"Loading Projector Weights from {projector_path}")
            if os.path.exists(projector_path):
                checkpoint = torch.load(projector_path, map_location="cpu", weight_only=False)
                projector_state = checkpoint
                self.projector.load_state_dict(projector_state)
                print("Projector Loaded Successfully :)")
            else:
                raise FileNotFoundError(f"Projector weights not found at {projector_path}")

            for params in self.projector.parameters():
                params.requires_grad = True

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

def build_stage2_optimizer(model):
  trainable_groups = [
      {
        "params": [p for n, p in model.llama_model.named_parameters() if p.requires_grad],
        "lr":2e-4
      },
      {
        "params":model.projector.parameters(),
        "lr":2e-5
      }
  ]
  return optim.AdamW(trainable_groups, weight_decay=0.0, fused=True)

if TRAINING_MODE:
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    TRAINING_CHECKPOINT = os.path.join(save_dir, "training_checkpoint.pt")


def save_checkpoint_stage2(model, optimizer, lr_schedular, scaler, epoch, loss, step):

    ADAPTER_PATH = os.path.join(save_dir, "lora_adapter")
    model.llama_model.save_pretrained(ADAPTER_PATH)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "optimizer_state_dict" : optimizer.state_dict(),
        "projector_state_dict" : model.projector.state_dict(),
        "lr_schedular_state_dict": lr_schedular.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }

    torch.save(checkpoint, TRAINING_CHECKPOINT)
    print("Saving Model Checkpoint....")
    print("CHECKPOINT SAVED.")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if TRAINING_MODE:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    image_processor = CLIPImageProcessor.from_pretrained(CLIP)
    tokenizer = AutoTokenizer.from_pretrained(LLM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(42)

    collate_fn = MultiTurnCollater(
    image_processor=image_processor,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
    )

    train_dataset = Custom50KMultiTurnDataset(
        data_list=custom_dataset
    )

    g = torch.Generator()
    g.manual_seed(42)
    shuffled_idx = torch.randperm(len(train_dataset), generator=g).tolist()
    total_batches = len(train_dataset) // BATCH_SIZE
    total_training_steps = (total_batches // ACCUMULATION_STEP) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * 0.03)

    model_stage2 = LLaVAStage2Model().to(device)
    w = model_stage2.projector.weight
    assert torch.isfinite(w).all() and torch.isfinite(model_stage2.projector.bias).all(), "projector has NaN/inf"
    print("proj weight absmax / mean:", w.abs().max().item(), w.abs().mean().item())

    # Enables Gradient Checkpointing
    model_stage2.clip_vision_model.eval()
    model_stage2.llama_model.train()
    model_stage2.projector.train()

    optimizer = build_stage2_optimizer(model_stage2)

    lr_schedular = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(MODEL_DTYPE == torch.float16))

    start_epoch = 0
    start_step = 0

    if os.path.exists(train_state_path) and os.path.exists(sd):
        # Load the LoRA Adapters
        res = set_peft_model_state_dict(model_stage2.llama_model, load_file(sd))
        print("unexpected:", res.unexpected_keys)
        print("LoRA B nonzero:", any(p.abs().sum() > 0 for n, p in model_stage2.llama_model.named_parameters() if "lora_B" in n))
        model_stage2.llama_model.train()

        # Load training variables and saved weights
        print(f"Loading Training Checkpoints from {train_state_path}")
        checkpoint = torch.load(train_state_path, map_location="cpu")
        model_stage2.projector.load_state_dict(checkpoint["projector_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_schedular.load_state_dict(checkpoint["lr_schedular_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint["step"]

        print(f"Model Checkpoint Restored...Resuming Training from Epoch: {start_epoch} and Step: {start_step}")
    else:
        print("NO Checkpoint Found. Starting Fresh Training. Trainable: LORA Adapter and Projector")

if TRAINING_MODE:
    remaining_indices = shuffled_idx[start_step * BATCH_SIZE:]
    active_dataset = Subset(train_dataset, remaining_indices)

    train_dataloader = DataLoader(
    active_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    generator=dataloader_generator,
    worker_init_fn = seed_worker,
    num_workers=6,
    persistent_workers=True,
    prefetch_factor=2
    )

    print("trainable:", {p.dtype for p in model_stage2.parameters() if p.requires_grad})  # want {float32}
    print("frozen:   ", {p.dtype for p in model_stage2.parameters() if not p.requires_grad})  # want {bfloat16}
    assert all(torch.isfinite(p).all() for p in model_stage2.parameters() if p.requires_grad)
    w0 = model_stage2.projector.weight.detach().clone()

    if ENABLE_TEST_DURING_TRAINING:
        #-----------Missing Image Test------------
        # Read Result: missing images should be zero. Ouput: 0[]
        missing = [it["image"] for it in tqdm(custom_dataset)
                if not os.path.exists(os.path.join(train_dataset.image_dir, os.path.basename(it["image"])))]
        print(len(missing), missing[:5])

        #---------------- Need of checkpointing test? ------------------#
        # Read Result: truncated at max_length(1024) should be = 0
        import numpy as np
        lens = []
        for i in range(0, len(custom_dataset), 50):          # every 50th sample
            convs = custom_dataset[i]["conversations"]
            n = sum(len(tokenizer(t["value"], add_special_tokens=False).input_ids) for t in convs)
            lens.append(n + 6 * len(convs))                   # ~6 header/eot tokens per turn
        lens = np.array(lens)
        print("p50/p90/p99:", np.percentile(lens, [50, 90, 99]).astype(int))
        print("truncated at 1024:", f"{(lens > 1024).mean():.1%}")

if TRAINING_MODE:
    for epoch in range(start_epoch, NUM_EPOCHS):
        running_loss = 0.0
        # Use train_dataloader to precent tqdm desychronisation during EPOCH change
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),initial=start_step if epoch == start_epoch else 0, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        optimizer.zero_grad()

        for step, batch in progress:
            current_step = (start_step if epoch == start_epoch else 0 ) + step
            with torch.amp.autocast(device, dtype=MODEL_DTYPE):
                output = model_stage2(
                pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                input_ids=batch["input_ids"].to(device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True)
                )
                loss = output.loss
                scaled_loss = loss / ACCUMULATION_STEP
            # NaN Gaurd to prevent nan posioning to weights
            if not torch.isfinite(loss):
                print(f"inf loss found at {current_step + 1}. skipping...")
                optimizer.zero_grad()
                continue

            #Backprogation
            running_loss += loss.item()
            scaler.scale(scaled_loss).backward()


            if (current_step + 1) % ACCUMULATION_STEP == 0:
                trainable_params = [p for p in model_stage2.parameters() if p.requires_grad]
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if torch.isfinite(gn):
                   scaler.step(optimizer)
                else:
                    print(f"non-finite grad norm at step {current_step}, skipping")
                scaler.update()
                lr_schedular.step()
                optimizer.zero_grad()

            if (current_step + 1) % PRINT_PER_STEP == 0:
                current_loss = loss.item()
                avg_loss = running_loss / PRINT_PER_STEP
                current_lr = lr_schedular.get_last_lr()[0]
                print(" ")
                print(f"Step: {current_step+1} | Loss: {current_loss:.4f} | Avg Rolling Loss: {avg_loss:.4f} | Current LR : {current_lr:.2e} | Nan Gaurd: {gn}")
                running_loss = 0.0

        # SMOKE TEST ---------------------------------------------------------------------------------------------------------------------------------
            if SMOKE_TEST and (current_step+1) % ACCUMULATION_STEP == 0:
                print(f"step {current_step+1} | loss {loss.item():.3f} | gn {gn.item():.3f} | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
            if SMOKE_TEST and current_step >= 23:
                break
        # SMOKE TEST ---------------------------------------------------------------------------------------------------------------------------------
            if (current_step + 1) % SAVE_PER_STEP == 0:
                save_checkpoint_stage2(
                    model=model_stage2,
                    optimizer=optimizer,
                    lr_schedular=lr_schedular,
                    scaler=scaler,
                    epoch=epoch,
                    step=current_step + 1,
                    loss=loss.item()
                )
                print(f"Checkpoint Save to local directory /wroking/llava_checkpoints at Step: {current_step + 1} ")
                if HUGGING_FACE_UPLOAD:
                    try:
                        hf_api.upload_file(
                            path_or_fileobj = "/workspace/llava_checkpoints/training_checkpoint.pt",
                            path_in_repo = "training_checkpoint.pt",
                            repo_id = MY_REPO,
                            repo_type = "dataset",
                            commit_message = f"Training Checkpoint Updated at {current_step+1}"
                        )
                        hf_api.upload_folder(
                            folder_path = "/workspace/llava_checkpoints/lora_adapter",
                            path_in_repo = "lora_adapter",
                            repo_id = MY_REPO,
                            repo_type="dataset",
                            commit_message = f"LoRA Adapter Updated at {current_step + 1}"
                        )
                    except Exception as e:
                        print("!Upload to hugging face failed", e)

    start_step = 0

    if not SMOKE_TEST:
        # Final Weights Save ..
        project_save_path = os.path.join(save_dir, "LLaVA Projector Final Weights.pt")
        torch.save(model_stage2.projector.state_dict(), project_save_path)

        NEW_ADAPTER_PATH = os.path.join(save_dir, "Final LoRA Adapters")
        model_stage2.llama_model.save_pretrained(NEW_ADAPTER_PATH)
        if HUGGING_FACE_UPLOAD:
            try:
                hf_api.upload_file(
                    path_or_fileobj = project_save_path,
                    path_in_repo = "LLaVA Final Projector Weights.pt",
                    repo_id = MY_REPO,
                    repo_type = "dataset",
                    commit_message = f"-- Final Projector Weight Uploaded."
                )
                hf_api.upload_folder(
                    folder_path = NEW_ADAPTER_PATH,
                    path_in_repo = "Final_LoRA Adapter Weights",
                    repo_id = MY_REPO,
                    repo_type="dataset",
                    commit_message = f"Final LoRA Adapter Uploaded"
                )
                print("Congratulations! YOU HAVE DONE IT.")
            except Exception as e:
                print("!Final Upload Fail. ", e)

#---------------------------------Mid training Image Recognition Test---------------------------------
if TRAINING_MODE and ENABLE_TEST_DURING_TRAINING:
    @torch.no_grad()
    def probe(model, batches, mode="real"):
        model.llama_model.eval()
        total = 0.0
        for b in batches:
            px = b["pixel_values"]
            if mode == "shuffled":
                px = px.roll(1, dims=0)          # wrong image, same text
            elif mode == "blank":
                px = torch.zeros_like(px)        # no visual content
            with torch.amp.autocast(device, dtype=MODEL_DTYPE):
                out = model(px,
                            b["input_ids"].to(device),
                            b["attention_mask"],
                            b["labels"])
            total += out.loss.item()
        model.llama_model.train()                # switch back for training; use .eval() if you're only testing
        return total / len(batches)

    # Read Result: there should be differnce between real, shuffled and blank. Less Value = Model Performed Better
    probe_idx = shuffled_idx[-100:]
    batches = [collate_fn([train_dataset[i] for i in probe_idx[j*4:(j+1)*4]]) for j in range(25)]
    for m in ["real", "shuffled", "blank"]:
        print(m, round(probe(model_stage2, batches, m), 3))

    #---------------------------------------Mid training Inference Test--------------------------------
    # Read Result: Compare the images with what does LLM outputs.
    from IPython.display import display

    @torch.no_grad()
    def chat(image, questions, max_new_tokens=120):
        m = model_stage2; m.eval()
        pv = image_processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device, dtype=MODEL_DTYPE)
        feats = m.clip_vision_model(pixel_values=pv, output_hidden_states=True).hidden_states[-2][:, 1:, :]
        img = m.projector(feats.float()).to(MODEL_DTYPE)
        emb = m.llama_model.get_input_embeddings()
        bos = emb(torch.tensor([[tokenizer.bos_token_id]], device=device))
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        history = ""
        for q in questions:
            history += f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ids = tokenizer(history, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            embeds = torch.cat([bos, img, emb(ids)], dim=1)
            out = m.llama_model.generate(
                inputs_embeds=embeds,
                attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long, device=device),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                eos_token_id=eot, pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            print(f"Q: {q}\nA: {ans}\n")
            history += f"{ans}<|eot_id|>"

    for fn in sorted(os.listdir(val_dir))[:5]:
        img = Image.open(os.path.join(val_dir, fn))
        display(img.resize((256, 256)))
        chat(img, ["What is in this image?", "What color is the most prominent object?", "Is there a person in the image?"])

    # Control: a plain gray image should not get a confident description
    chat(Image.new("RGB", (336, 336), (128, 128, 128)), ["What is in this image?"])



    with torch.no_grad():
        px = batches[0]["pixel_values"].to(device, dtype=MODEL_DTYPE)
        f = model_stage2.clip_vision_model(pixel_values=px, output_hidden_states=True).hidden_states[-2][:, 1:, :]
        print("shape:", tuple(f.shape))                         # (4, 576, 1024)
        print("feature diff between images:", (f[0] - f[1]).abs().mean().item())   # should be clearly > 0

    #------------------------GPU Limit Test to set BATCH_SIZE and ACCUMULATION_STEP accordingly-------------------
    # Read Result: worst-case peak should be within gpu vram limits. Boundry of 2-4 GB is best
    torch.cuda.reset_peak_memory_stats()
    B = BATCH_SIZE
    dummy = dict(
        pixel_values=torch.randn(B, 3, 336, 336),
        input_ids=torch.randint(1000, 5000, (B, MAX_LENGTH)),
        attention_mask=torch.ones(B, MAX_LENGTH, dtype=torch.long),
        labels=torch.randint(1000, 5000, (B, MAX_LENGTH)),
    )
    with torch.amp.autocast(device, dtype=MODEL_DTYPE):
        loss = model_stage2(**dummy).loss
    loss.backward()
    print(f"worst-case peak: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    optimizer.zero_grad(set_to_none=True)

