# :robot: femtoGPT

![crates.io](https://img.shields.io/crates/v/femto-gpt.svg)
![GitHub top language](https://img.shields.io/github/languages/top/keyvank/femtoGPT)
![GitHub](https://img.shields.io/github/license/keyvank/femtoGPT)

femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer.

It can be used for both *inference* and *training* of GPT-style language-models
using **CPUs** and **GPUs**!

## Intro

Everything is implemented from scratch, including the tensor processing logic
along with training/inference code of a minimal GPT architecture.

The architecture is very similar/almost identical with Andrej Karpathy's
[nanoGPT video lecture](https://github.com/karpathy/ng-video-lecture).

femtoGPT is a great start for those who are fascinated by LLMs and would like to
understand how these models work in very deep levels.

femtoGPT uses nothing but random generation libraries (`rand`/`rand-distr`), data-serialization
libraries (`serde`/`bincode` for saving/loading already trained models) and a
parallel computing library (`rayon`).

femtoGPT is ~~EXTREMELY SLOW~~ ***relatively fast on CPU 😉***, and most of the
primitive operations (E.g Matrix multiplication) are implemented in the simplest way possible.

Correctness of gradients is checked using gradient-check method, though it still is very
possible that some layers are implemented wrongly.

([Discord server](https://discord.gg/wTJFaDVn45) for discussions around the project!)

## Usage

Make sure you have the Rust toolchain on your system, in order to compile and run
the project:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

If you want to train using a GPU, you will first need to make sure your GPU drivers
are correctly installed on your system, and their OpenCL runtimes are available.

On Debian systems, you can setup OpenCL runtimes by installing the package `ocl-icd-opencl-dev`:

`sudo apt install ocl-icd-opencl-dev`

***GOOD NEWS!*** *Since femtoGPT's GPU implementation is based on OpenCL, it can
run on both NVIDIA and AMD cards, and you won't need to install heavy-weight
CUDA-toolkits on your system. OpenCL runtimes would suffice!*

Now you'll just need to put the text you want to train your GPT model on, inside
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

**UPDATE 5th June 2023:**

This has been a new output, after more hours of training on a model with similar scale:

```
What like but wore pad wo me che nogns yous dares,
As supt it nind bupart 'the reed:
And hils not es
```

Obviously the model has started to learn some words and punctuation rules!

**UPDATE 9th June 2023:**

Model was able to reach loss value of ~1.4

Here is an example output:

```
Adistition gone; true; schistoes for mine souls!
Before your home, bariechts should be
Carlam on that's a worf quirer of him so.
What look'd lack away more
To him foot; one hour fortious of saves:
Son;
'Tis all Earl mmistling me.

HARSARTIO:
Why, idless of my mocks fan that were percious.
Having I will thou should and the muour ne'er shor
To purple, when duke thy are out him.
But will bid you doth remember nature.
Even OF hencomey, carniffeit, I joy
Warming; my deed, but it on of mortard,
```

**UPDATE 10th June 2023, sparks of AGI? LOL**

Having trained a 5-layer model on a dataset gathered from Reddit, tokenized with sentencepiece tokenizer with a vocab-size of 500, for about ~10 hours on a 32-core CPU:

Prompt **"I think"**:

```
I think it s not a scientific method of atheism
I think it s not a fairly simple possible to have a
I think that s a possible point, but I m not sure if you re not
I think that s a great idea, but I m not sure how these are you talking
I think it s a simple thing to do, but I m not sure how thes
I think that s a good thing, but I don t think that s what I mean, I think I
I think that s the only way to be able to use it to be in the same way, but
I think that the police are relatively relatively relative
I think that s a simple moron is a simple character
I think that s the only way to go to the second side of the pol
```

Prompt **"Life is"**:

```
Life is not a good idea, but it s not a perfection of the opp
Life is a good thing to say that I don t know what you re talking about, but I don
Life is not the same as a religion, but I m not sure if you re a
Life is a perfectly good job of arguing that you are alm
Life is a perfectly good job of the opposite of the f
Life is a fundamentalist, and then I m not sure how the h
Life is not a good idea, and it s not a perfectly good job, but I
Life is not the same as atheists, but that s the only way to be ac
Life is a bit of a single one of these industry is a f
Life is a good idea to get the opposite of the police offic
```

Prompt **"So sad that"**:

```
So sad that you can tell you what? I think I ve been using it on the scre
So sad that I don t know about it, but I don t think I m not afraid to
So sad that I m not sure if you re not arguing with the fact that you
So sad that I was involved in the future, and I have a few we
So sad that s what I said, I m sure you are almost everything you
So sad that you can do it, and I don t think that the fact that it s a po
So sad that I m not sure if you re arguing with the fact that they are
So sad that s the one too much time, but I m not sure if you re arg
So sad that you are sadly supposed to be a big deal in the world
So sad that I don t know about this, but I m not sure how you can do it, but
```

**UPDATE 29th June 2023**

After the implementation of the GPU trainer, we were able to train larger models. 
Here are some samples from a 8-layer 8-head 128-embedding-degree model, trained on
TinyStories dataset on a vocab-size of 1000:

```
Once upon a time, there was a little girl named Lily.
She loved to play with her toys and she had a lot of fun.
One day, Lily saw a big chicky playing with her toys.
She asked her mom, "Can I play with her toys?" Her mom said,
"Sure, Lily. But we have to clean the pales. Let's suet some candy, Lily."
Lily nodded and went to her mom. They played with the mots and staugning her toys.  
```

```
Once upon a time, there was a little girl named Lily.
She loved to play outside and explore. One day, she found a jung on the ground.
She picked it up and tecked it. She ran around and saw it. She was very sad.
She asked her mom for her mom. Her mom said, "Lily, I'm going to find it!" Lily said.
She ran to the slock and took her to the teplace. She went to the park and found a molla.
```

```
There was a boy named Tim. Tim loved to play with his toys.
One day, Tim's mom came to the park. Tim saw a big, red ball and wanted to play with it.
Tim wanted to play with the ball. Tim was very excited. He wanted to play with the ball.
But the ball was too fast. Tim wanted to play with the ball. But the ball was too fast.
Tim tried to catch it, but it was too fast. Tim was sad. He tried to run away,
but he did not want to play. Tim was sad. He did not want to play with the ball.
```
