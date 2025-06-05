import numpy as np
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = ((predictions - labels) ** 2).mean()
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(labels)
    return {"rmse": rmse, "r2": r2}


def train(model: nn.Module, train_dataset: Dataset, val_dataset: Dataset):
    output_dir = "./results_ruBERT"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        learning_rate=6e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,    
        save_total_limit=5,  
        num_train_epochs=25,
        weight_decay=0.1,
        warmup_ratio=0.05,
        max_grad_norm=1.5,
        lr_scheduler_type="linear",
        logging_dir=f"./{output_dir}/logs",
        bf16=torch.cuda.is_bf16_supported(), 
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="r2", 
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    checkpoint = get_last_checkpoint(output_dir)
    trainer.train()
