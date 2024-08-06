import logging
from random import randint, choices

import fire
import pyrootutils
from dotenv import load_dotenv
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


def generate_data(
    num_samples: int = 100_000,
    hla_seq_len: int = 40,
    peptide_seq_len: int = 10,
    vocab_string: str = "ARNDCEQGHILKMFPSTWYV",
    seed: int = randint(0, 2**32 - 1),
):
    
    set_all_seeds(seed)
    
    vocab = list(vocab_string)
    hla_seqs = []
    peptide_seqs = []
    labels = []
    
    for _ in tqdm(range(num_samples)):

        hla_seq_len_i = randint(35, hla_seq_len)
        peptide_seq_len_i = randint(8, peptide_seq_len)

        hla_seq = "".join(choices(vocab, k=hla_seq_len_i))
        peptide_seq = "".join(choices(vocab, k=peptide_seq_len_i))
        label = 1 if hla_seq[0] == peptide_seq[0] else 0
        # label = 1 if "A" in peptide_seq else 0
        
        hla_seqs.append(hla_seq)
        peptide_seqs.append(peptide_seq)
        labels.append(label)
    
    with open(f"{PROJECT_ROOT}/data/seqs.txt", "w") as f:
        f.write("HLA_SEQ,PEPTIDE_SEQ,LABEL\n")
        for hla_seq, peptide_seq, label in zip(hla_seqs, peptide_seqs, labels):
            f.write(f"{hla_seq},{peptide_seq},{label}\n")

if __name__ == "__main__":
    fire.Fire(generate_data)