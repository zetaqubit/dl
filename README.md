# dl
Implementations of different Deep Learning architectures and algorithms, using only basic ops provided by pytorch.

# TODOs

## transformer

- [x] Implement transformer with self-attention
- [ ] Add support for cross-attention, as used in NMT

## rnn

- [ ] Implement RNN
- [ ] Implement LSTM

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
