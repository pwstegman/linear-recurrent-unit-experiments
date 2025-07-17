# %%
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

type(enc.decode([1, 2, 3]))
