"""Train models."""

import logging
from random import randint

import fire
import pyrootutils
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from load_data import PeptideTokenizer, collate_fn, get_dataset
from model import (
    LSTMPeptideClassifier,
    Mamba2PeptideClassifier,
    SRNPeptideClassifier,
    TransformerPeptideClassifier,
)
from torch import optim
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy, BinaryF1Score
from tqdm import tqdm
from utils.utils import get_logger, set_all_seeds

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()


def main(
    seed: int = randint(0, 2**32 - 1),
    # Model parameters
    model_type: str = "transformer",
    num_layers: int = 6,
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    pooling_dimension: int = 1,
    input_size: int = 512,
    hidden_size: int = 2048,
    bidirectional: bool = False,
    # Training parameters
    batch_size: int = 4,
    dataset_prop: float = 1.0,
    base_lr: float = 1e-4,
    # Dataset parameters
    data_file: str = "seqs.txt",
):
    set_all_seeds(seed)

    dataset = get_dataset(sample_ratio=dataset_prop, filename=data_file)
    tokenizer = PeptideTokenizer()

    log.info(dataset)

    model: TransformerPeptideClassifier | SRNPeptideClassifier | Mamba2PeptideClassifier

    if model_type == "transformer":
        model = TransformerPeptideClassifier(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=len(tokenizer),
            pooling_dimension=pooling_dimension,
        )
    elif model_type == "srn":
        model = SRNPeptideClassifier(
            dropout=dropout,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=len(tokenizer),
            bidirectional=bidirectional,
            pooling_dimension=pooling_dimension,
        )
    elif model_type == "lstm":
        model = LSTMPeptideClassifier(
            dropout=dropout,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=len(tokenizer),
            bidirectional=bidirectional,
            pooling_dimension=pooling_dimension,
        )
    elif model_type == "mamba2":
        model = Mamba2PeptideClassifier(
            d_model=512,
            n_layer=num_layers,  # number of Mamba-2 layers in the language model
            d_state=128,  # state dimension (N)
            d_conv=4,  # convolution kernel size
            expand=2,  # expansion factor (E)
            headdim=64,  # head dimension (P)
            chunk_size=64,  # matrix partition size (Q)
            vocab_size=len(tokenizer),
            pad_vocab_size_multiple=16,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    log.info(model)
    log.info(f"Number of parameters: {model.num_parameters}")

    if model_type == "mamba2":
        padded_collate_rn = lambda data: collate_fn(data, seq_len_multiple=64)  # noqa: E731
    else:
        padded_collate_rn = collate_fn

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=padded_collate_rn,
    )
    val_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=padded_collate_rn,
    )

    num_steps = len(train_dataloader)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    accelerator = Accelerator()
    log.info(accelerator.state)

    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )

    metrics = {
        "binary_accuracy": BinaryAccuracy(),
        "binary_f1": BinaryF1Score(),
    }

    model.train()
    for batch in (pbar := tqdm(train_dataloader)):
        optimizer.zero_grad()
        inputs = batch["input_ids"]
        targets = batch["labels"]
        predictions = model(inputs)
        targets_for_loss = targets.squeeze()
        predictions_for_loss = predictions.squeeze()

        loss = F.cross_entropy(
            input=predictions_for_loss,
            target=targets_for_loss,
        )
        pbar.set_postfix({"loss": loss.item()})
        accelerator.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0)

        optimizer.step()
        scheduler.step()

        # Update metrics
        for metric in metrics.values():
            metric.update(
                input=predictions_for_loss.argmax(dim=1),
                target=targets_for_loss,
            )

    print("Mean training accuracy: ", metrics["binary_accuracy"].compute().item())
    print("Mean training f1: ", metrics["binary_f1"].compute().item())
    for metric in metrics.values():
        metric.reset()

    model.eval()
    for batch in tqdm(val_dataloader):
        inputs = batch["input_ids"]
        targets = batch["labels"]

        predictions = model(inputs)

        # Update metrics
        for metric in metrics.values():
            metric.update(
                input=predictions_for_loss.argmax(dim=1),
                target=targets_for_loss,
            )

    print("Mean validation accuracy: ", metrics["binary_accuracy"].compute().item())
    print("Mean validation f1: ", metrics["binary_f1"].compute().item())


if __name__ == "__main__":
    fire.Fire(main)
