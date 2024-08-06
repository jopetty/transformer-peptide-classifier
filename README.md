# transformer-peptide-classifier

Let's classify some sequences.

## Setup

1. Install `uv`, a python virtual environment manager (think Conda but infinitely better): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone repo.
3. `cd` into repo and create virtual environment: `uv venv`
4. Activate virtual environment: `source .venv/bin/activate`
5. Install all dependencies for project: `uv pip install -e .`

## Use

I don't know exactly what the format of the data is like, but I took a guess. You can generate some sample data using `python src/generate_data.py`. This will create a file called `seqs.txt` inside the `data/` directory. Right now, it labels each sequence pair by whether or not the peptide sequence has `A` in it.

You can train models with `python src/train.py`. For detailed usage, look at the arguments to the `main` function there; each value is configurable from the command line with a flag:

```bash
python src/train.py --model_type=transformer  # trains a transformer
python src/train.py --model_type=lstm  # trains an LSTM
```

Everything should "just work" on multiple GPUs w/ accelerate.
