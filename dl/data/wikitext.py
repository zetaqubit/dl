"""Processing wikitext to pages."""

import datasets as hf_datasets  # huggingface datasets
import pandas as pd

def lines_to_pages(hf_ds):
  ds_out = {}
  for split, ds in hf_ds.items():
    pages = []
    page = []
    for ex in ds:
      line = ex['text']
      if line.startswith(' = ') and not line.startswith(' = = '):
        # New page with <h1>.
        if len(page) > 0:
          pages.append(''.join(page))
        page = [line]
        continue
      if len(line) == 0:
        continue
      if line.startswith(' = '):
        page.append('\n')  # extra newline before headings for readability.
      # normalize weird @ @ symbols.
      line = line.replace(' @-@ ', '-')
      line = line.replace(' @.@ ', '.')
      line = line.replace(' @,@ ', ',')
      # make punctuation conventional.
      line = line.replace(' , ', ', ').replace(' . ', '. ').replace(" '", "'")
      line = line.replace(' : ', ': ').replace(' ; ', '; ')
      line = line.replace('( ', '(').replace(' )', ')')
      page.append(line)
    df = pd.DataFrame(pages, columns=['text'])
    ds_out[split] = hf_datasets.Dataset.from_pandas(df, split=split)
  ds_out = hf_datasets.DatasetDict(ds_out)
  return ds_out
