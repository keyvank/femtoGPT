# femtoGPT

femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer.

Everything is implemented from scratch, including the tensor processing logic
along with training/inference code of a minimal GPT architecture.

The architecture is very similar/almost identical with Andrej Karpathy's
![nanoGPT](https://github.com/karpathy/nanoGPT) video lecture.

femtoGPT is a great start for those who are fascinated by LLMs and would like to
understand how these models work in very deep levels.

femtoGPT uses nothing but a random generation library (`rand`), data-serialization
libraries (`serde`/`bincode` for saving/loading already trained models) and a 
parallel computing library (`rayon`).
