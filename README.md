# dl
Implementations of different Deep Learning architectures and algorithms, using only basic ops provided by pytorch.

# Experiment Results

## transformer

| Model | wikitext-103 ppl | Closest public model |
| ------- | --------- | ---- |
| gpt2 12l | [26.7](https://tensorboard.dev/experiment/1J35Jwg0RlSpOwVKgkqL4g/#scalars) | [26.37 (gpt2-medium)](https://paperswithcode.com/sota/language-modelling-on-wikitext-103?metric=Validation%20perplexity)

## toy datasets

* RNN vs LSTM vs GRU on toy dataset of "abcdef...": [tensorboard](https://tensorboard.dev/experiment/7iXNkEoKSNegP4D2Sdenpw/#scalars)
* RNN vs LSTM vs GRU on toy dataset of "a...ab..bc..c...": [tensorboard](https://tensorboard.dev/experiment/jy94YBDhQ5G82msYznREVA/#scalars)


# TODOs

## transformer

- [x] Implement transformer with self-attention
- [x] Implement sinusoidal position embeddings
- [x] Implement relative position bias a la T5
- [x] Implement RoPE embeddings
- [x] Add support for cross-attention, as used in NMT
- [ ] Implement beam search decoding

## rnn

- [x] Implement RNN
- [x] Implement LSTM
- [x] Implement GRU
- [ ] Implement RWKV

## examples/wikitext

- [x] Load wikitext dataset
- [x] Implement training loop
- [x] Implement tool for generating text
- [x] Set up tensorboard metrics, text samples
- [x] Implement model checkpoint saving / resume
- [x] Init correctly and verify initial loss is -log(1/50000)
- [x] Limit train set to 1 batch and verify train loss goes to 0
- [ ] Try mixed precision
- [ ] Scale to 1.5B param model
- [ ] Scale to 1024 sequence length

## eval

- [ ] Implement evaluation framework
- [ ] Collect popular LM benchmarks and published metrics