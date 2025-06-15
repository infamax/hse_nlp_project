import argparse
import torch
import os
import torch.nn as nn
import yaml


from transformers import AutoTokenizer

from models.nn.dataset import AutoRUDataset
from models.nn.models import TransformerRegression

from tqdm import tqdm
from utils.consts import SEED
from utils.set_seed import set_seed


def _get_criterion(criterion_name: str) -> nn.Module:
    if criterion_name.lower() == "mse":
        return nn.MSELoss()
    elif criterion_name.lower() == "mae":
        return nn.L1Loss()
    raise ValueError(f"Get unsupported criterion for this task")


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    batch_size: int,
    experiment_name: str,
    device: str,
    path_to_save_model_best_weights: str,
) -> tuple[nn.Module, list[float], list[float]]:
    best_val_loss = float("inf")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    train_epoch_losses = []
    val_epoch_losses = []
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        print(f"Epoch: {epoch}")

        # Цикл обучения по обучающей выборке из датасета
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc="Training"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids.squeeze(), attention_mask.squeeze())

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_epoch_losses.append(epoch_loss)

        # Прогон модели по валидационной выборке
        model.eval()
        val_running_loss = 0.0

        for data in tqdm(val_loader, desc="Validation"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            with torch.no_grad():
                outputs = model(input_ids.squeeze(), attention_mask.squeeze())
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * input_ids.size(0)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_losses.append(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            print(f"New best model on epoch: {epoch}")
            best_val_loss = val_epoch_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    path_to_save_model_best_weights, f"{experiment_name}_best_model.pth"
                ),
            )

        print(f"Train loss: {epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}")
    return model, train_epoch_losses, val_epoch_losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-folder",
        help="Folder that contain a train and val parts of dataset",
        type=str,
    )
    parser.add_argument("-c", "--config", help="Experiment config path", type=str)
    args = parser.parse_args()
    dataset_folder, config_path = args.dataset_folder, args.config

    if not os.path.exists(config_path):
        print(
            "Error! Cannot run experiment. Model config file doesn't exist. Check your path to config correct"
        )
        return -1

    path_to_train_dataset = os.path.join(dataset_folder, "train/train.parquet")
    path_to_val_dataset = os.path.join(dataset_folder, "val/val.parquet")

    if not os.path.exists(path_to_train_dataset):
        print(
            f"Error! Cannot run experiment. Train dataset file doesn't exist: {path_to_train_dataset}"
        )
        return -1

    if not os.path.exists(path_to_val_dataset):
        print(
            f"Error! Cannot run experiment. Val dataset file doesn't exist: {path_to_val_dataset}"
        )
        return -1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(SEED)

    model_name = config["model_name"]
    dropout_rate = config["dropout_rate"]
    max_len = config["max_len"]
    text_splitter = config["text_splitter"]
    experiment_name = config["experiment_name"]
    criterion_name = config["criterion_name"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    path_to_save_model_best_weights = config["path_to_save_model_best_weights"]
    os.makedirs(path_to_save_model_best_weights, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = AutoRUDataset(
        path_to_train_dataset, tokenizer, max_len, text_splitter
    )
    val_dataset = AutoRUDataset(path_to_val_dataset, tokenizer, max_len, text_splitter)
    model = TransformerRegression(model_name, dropout_rate)
    criterion = _get_criterion(criterion_name)
    optimizer = torch.optim.Adam(model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for param in model.parameters():
        param.data = param.data.contiguous()
    model = model.to(device)

    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        experiment_name=experiment_name,
        device=device,
        path_to_save_model_best_weights=path_to_save_model_best_weights,
    )

    return 0


if __name__ == "__main__":
    exit(main())
