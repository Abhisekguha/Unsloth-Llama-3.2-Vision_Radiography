Follow the steps:

# !pip install unsloth
# !pip install transformers datasets huggingface-hub

# Connect the huggingface acc.

from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_TOKEN"])

# import the necessary libraries 

import os 
from unsloth import FastVisionModel
import torch 
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig 

# 1. load the model
"unsloth/Llama-3.2-11B-Vision-Instruct"

# 2. load dataset and instruction mentioned in the code
dataset = load_dataset("unsloth/Radiology_mini", split = "train")

# 3. parameters and key observations before training
# 4. Training -- FastVisionModel, arg = SFTConfig

args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
)

initial trainig loss: 3.680100
final training loss: 1.164900



# 5. After trainig performance
# 6. save model in huggingface
