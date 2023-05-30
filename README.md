# femtoGPT

femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer.

Everything is implemented from scratch, including the tensor processing logic
along with training/inference code of a minimal GPT architecture.

The architecture is very similar/almost identical with Andrej Karpathy's
![nanoGPT video lecture](https://github.com/karpathy/ng-video-lecture).

femtoGPT is a great start for those who are fascinated by LLMs and would like to
understand how these models work in very deep levels.

femtoGPT uses nothing but a random generation library (`rand`), data-serialization
libraries (`serde`/`bincode` for saving/loading already trained models) and a
parallel computing library (`rayon`).

## Output samples

After hours of training on the Shakespeare database, on a 300k parameter model,
this has been the output:

```
LIS:
Tore hend shater sorerds tougeng an herdofed seng he borind,
Ound ourere sthe, a sou so tousthe ashtherd, m se a man stousshan here hat mend serthe fo witownderstesther s ars at atheno sel theas,
thisth t are sorind bour win soutinds mater horengher
```

This is embarrassingly bad, but looking at the bright side, it seems like it has
been able to generate words that are easy to pronounce.

I'm currently training a 10M parameter model to further examine the correctness
of my implementation.
