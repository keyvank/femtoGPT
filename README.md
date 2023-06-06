# :robot: femtoGPT

femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer.

Everything is implemented from scratch, including the tensor processing logic
along with training/inference code of a minimal GPT architecture.

The architecture is very similar/almost identical with Andrej Karpathy's
[nanoGPT video lecture](https://github.com/karpathy/ng-video-lecture).

femtoGPT is a great start for those who are fascinated by LLMs and would like to
understand how these models work in very deep levels.

femtoGPT uses nothing but random generation libraries (`rand`/`rand-distr`), data-serialization
libraries (`serde`/`bincode` for saving/loading already trained models) and a
parallel computing library (`rayon`).

femtoGPT is ***EXTREMELY SLOW***, since most of the primitive operations (E.g Matrix multiplication)
are implemented in the simplest way possible.

Correctness of gradients is checked using gradient-check method, though it still is very
possible that some layers are implemented wrongly. (E.g I'm not sure if my LayerNorm is
bug-free?)

**HELP!** *IF YOU HAVE A COMPUTER WITH PLENTY OF CPUS AND YOU DON'T MIND RUNNING femtoGPT
FOR A FEW HOURS/DAYS, YOU CAN HELP THIS PROJECT A GREAT DEAL! PLZ CONTACT ME :)*

([Discord server](https://discord.gg/wTJFaDVn45) for discussions around the project!)

## Usage

Easy! You'll just need to put the text you want to train your GPT model on, inside
`dataset.txt`. Make sure it has a small number of unique characters! (E.g. the
current dataset has only used 65 different unique characters!)

Then you'll need to run:

```
cargo run --release
```

It will start training the model and will put the training data in the `train_data`
directory. You can stop the training and continue later!

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

**UPDATE:**

This has been a new output, after more hours of training on a model with similar scale:

```
What like but wore pad wo me che nogns yous dares,
As supt it nind bupart 'the reed:
And hils not es
```

Obviously the model has started to learn some words and punctuation rules!
