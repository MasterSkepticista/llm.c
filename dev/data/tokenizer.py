"""Writes GPT-2 vocabulary to a binary file."""
import struct
import tiktoken
import numpy as np

def main():
  tokenizer = tiktoken.get_encoding("gpt2")
  filename = "gpt2_tokenizer.bin"
  num_tokens = tokenizer.max_token_value + 1
  header = np.zeros(256, np.int32)
  header[0] = 20240328
  header[1] = 2  # 2 includes eot_token
  header[2] = num_tokens  # 50256 + 1
  header[3] = tokenizer.eot_token
  with open(filename, "wb") as f:
    f.write(header.tobytes())
    for i in range(num_tokens):
      b = tokenizer.decode_bytes([i])
      length = len(b)
      assert length < 256
      f.write(struct.pack("<B", length))  # little-endian order
      f.write(b)

  print(f"Written {filename} successfully.")

if __name__ == "__main__":
  main()