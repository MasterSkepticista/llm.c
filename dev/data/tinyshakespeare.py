"""Builds a Tiny Shakespeare GPT-2 tokenized dataset."""
import os
import requests
import tiktoken

import numpy as np

FILE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def write_datafile(tokens: list, filename: str):
  tokens = np.asarray(tokens)
  assert (0 <= tokens).all() and (tokens < 2**16).all(), "Token dict too large for u16 type."
  tokens = tokens.astype(np.uint16)
  # Header, to match original impl.
  header = np.zeros(256, dtype=np.int32)
  header[0] = 20240520
  header[1] = 1
  header[2] = len(tokens)  # Tokens start after 256*4 header bytes.
  size = len(header) * 4 + len(tokens) * tokens.itemsize
  print(f"Writing {len(tokens):,} tokens to {filename}, total {size:,} bytes")
  with open(filename, "wb") as f:
    f.write(header.tobytes())
    f.write(tokens.tobytes())

def main():
  # Download textfile.
  response = requests.get(FILE_URL)
  if response.status_code == 200:
    text = response.content.decode()
    print("File downloaded successfully.")
  else:
    print("Download failed (response code: %d)", response.status_code)
  
  # Tokenize.
  enc = tiktoken.get_encoding("gpt2")
  encode = lambda s: enc.encode_ordinary(s)
  eot = enc._special_tokens["<|endoftext|>"]
  sections = text.split("\n\n")
  tokens = []
  for i, s in enumerate(sections):
    tokens.append(eot)
    s_pad = s
    if i != len(sections) - 1:
      s_pad = s + "\n\n"
    tokens.extend(encode(s_pad))

  # Split train/test.
  train_tokens, val_tokens = tokens[32768:], tokens[:32768]

  # Write to bin.
  out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyshakespeare")
  os.makedirs(out_dir, exist_ok=True)
  write_datafile(train_tokens, os.path.join(out_dir, "tiny_shakespeare_train.bin"))
  write_datafile(val_tokens, os.path.join(out_dir, "tiny_shakespeare_val.bin"))

if __name__ == "__main__":
  main()