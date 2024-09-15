"""Load and process data."""

import logging
import math
from enum import StrEnum
from functools import partial
from random import randint

import fire
import pyrootutils
import torch
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from torch.nn import functional as F  # noqa: N812
from transformers import PreTrainedTokenizerFast
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


AMINO_ACID_VOCAB = "ARNDCEQGHILKMFPSTWYV"


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer."""

    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def as_dict(cls):
        """Return the special token as a dictionary."""
        return dict(
            sorted(
                {v: i for i, v in enumerate(cls.values())}.items(), key=lambda x: x[1]
            )
        )

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)


class PeptideTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        model_max_length: int = 512,
        # split_on_whitespace: bool = True,
        **kwargs,
    ):
        """Character tokenizer for Hugging Face transformers.

        Args:
            model_max_length (int): Model maximum sequence length.

            split_on_whitespace (bool): Include a Whitespace pre-tokenizer.
        """

        vocab_dict = SpecialTokens.as_dict() | {
            ch: i + len(SpecialTokens.values()) for i, ch in enumerate(AMINO_ACID_VOCAB)
        }
        tokenizer_base = Tokenizer(
            WordLevel(vocab=vocab_dict, unk_token=SpecialTokens.UNK)
        )
        tokenizer_base.post_processor = TemplateProcessing(
            # single=f"{SpecialTokens.BOS} $A",
            pair=f"{SpecialTokens.BOS} $A {SpecialTokens.SEP} $B {SpecialTokens.EOS}",
            special_tokens=[
                (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS)),
                (SpecialTokens.SEP, tokenizer_base.token_to_id(SpecialTokens.SEP)),
                (SpecialTokens.EOS, tokenizer_base.token_to_id(SpecialTokens.EOS)),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer_base,
            bos_token=SpecialTokens.BOS,
            eos_token=SpecialTokens.EOS,
            unk_token=SpecialTokens.UNK,
            pad_token=SpecialTokens.EOS,  # use [EOS] as [PAD]
            mask_token=SpecialTokens.MASK,
            sep_token=SpecialTokens.SEP,
            cls_token=SpecialTokens.CLS,
            model_max_length=model_max_length,
            padding_side="left",
            **kwargs,
        )

        self.add_tokens(list(AMINO_ACID_VOCAB))


def preprocess(
    examples: DatasetDict,
    tokenizer: PeptideTokenizer,
):
    return tokenizer(
        examples["HLA_SEQ"],
        examples["PEPTIDE_SEQ"],
        return_token_type_ids=False,
        return_attention_mask=False,
    )


def collate_fn(data, seq_len_multiple: int | None = None):
    labels = []
    input_ids = []
    for item in data:
        labels.append(torch.tensor(item["LABEL"], dtype=torch.float32))
        input_ids.append(torch.tensor(item["input_ids"], dtype=torch.int32))

    # Pad input_ids if necessary
    seq_len = 0
    for item in input_ids:
        seq_len = max(seq_len, item.shape[0])
        # Ensure that the sequence length is a multiple of seq_len_multiple

    if seq_len_multiple is not None:
        # print("div:", seq_len // seq_len_multiple)
        # print("ceil:", math.ceil(seq_len / seq_len_multiple))
        seq_len = math.ceil(seq_len / seq_len_multiple) * seq_len_multiple

    # print("seq_len:", seq_len)
    # print("seq_len_multiple:", seq_len_multiple)

    for i in range(len(input_ids)):
        if input_ids[i].shape[0] < seq_len:
            input_ids[i] = F.pad(
                input_ids[i],
                (0, seq_len - input_ids[i].shape[0]),
                value=SpecialTokens.EOS.index,
            )

    labels = torch.stack(labels, dim=0).unsqueeze(1)
    input_ids = torch.stack(input_ids, dim=0)

    # print(labels.shape, input_ids.shape)
    # raise SystemExit

    return {
        "labels": labels,
        "input_ids": input_ids,
    }


def get_dataset(
    seed: int = randint(0, 2**32 - 1),
    sample_ratio: float = 1.0,
    filename: str = "seqs.txt",
):
    set_all_seeds(seed)

    sample_ratio = min(1.0, max(0.0, sample_ratio))

    log.info(f"Loading dataset from {filename}")
    dataset = load_dataset(
        "csv", data_files=f"{PROJECT_ROOT}/data/{filename}", split="train"
    )

    dataset = dataset.select(range(int(sample_ratio * len(dataset)))).train_test_split(
        test_size=0.2
    )

    tokenizer = PeptideTokenizer()

    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
    )
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["HLA_SEQ", "PEPTIDE_SEQ"],
    )

    return dataset


if __name__ == "__main__":
    fire.Fire(get_dataset)
