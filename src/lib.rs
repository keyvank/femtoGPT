//! **femtoGPT** is a pure Rust implementation of a minimal Generative Pretrained Transformer.
//!
//! It can be used for both *inference* and *training* of GPT-style language-models
//! using **CPUs** and **GPUs**!
//!
//!
//! ## Usage
//!
//! Training:
//!
//!`cargo run --release -- train`
//!
//! Inference:
//!
//! `cargo run --release -- infer`
//!
//! (Note: Add `--features gpu` in order to leverage GPU speedups!)
//!
//! ## Intro
//!
//! Everything is implemented from scratch, including the tensor processing logic
//! along with training/inference code of a minimal GPT architecture.
//!
//! The architecture is very similar/almost identical with Andrej Karpathy's
//![nanoGPT video lecture](https://github.com/karpathy/ng-video-lecture).
//!
//! femtoGPT is a great start for those who are fascinated by LLMs and would like to
//! understand how these models work in very deep levels.
//!
//! femtoGPT uses nothing but random generation libraries (`rand`/`rand-distr`), data-serialization
//! libraries (`serde`/`bincode` for saving/loading already trained models) and a
//! parallel computing library (`rayon`).
//!
//! femtoGPT is ~~EXTREMELY SLOW~~ ***relatively fast on CPU 😉***, and most of the
//! primitive operations (E.g Matrix multiplication) are implemented in the simplest way possible.
//!
//! Correctness of gradients is checked using gradient-check method, though it still is very
//! possible that some layers are implemented wrongly.
//!
//! ([Discord server](https://discord.gg/wTJFaDVn45) for discussions around the project!)
//!
//! ## Usage
//!
//! Make sure you have the Rust toolchain on your system, in order to compile and run
//! the project:
//!
//! `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
//!
//! If you want to train using a GPU, you will first need to make sure your GPU drivers
//! are correctly installed on your system, and their OpenCL runtimes are available.
//!
//! On Debian systems, you can setup OpenCL runtimes by installing the package `ocl-icd-opencl-dev`:
//!
//! `sudo apt install ocl-icd-opencl-dev`
//!
//! ***GOOD NEWS!*** *Since femtoGPT's GPU implementation is based on OpenCL, it can
//! run on both NVIDIA and AMD cards, and you won't need to install heavy-weight
//! CUDA-toolkits on your system. OpenCL runtimes would suffice!*
//!
//! Now you'll just need to put the text you want to train your GPT model on, inside
//! `dataset.txt`. Make sure it has a small number of unique characters! (E.g. the
//! current dataset has only used 65 different unique characters!)
//!
//! Then you'll need to run:
//!
//! ```example
//! cargo run --release
//! ```
//!
//!It will start training the model and will put the training data in the `train_data`
//! directory. You can stop the training and continue later!
//! 
pub mod funcs;
pub mod gpt;
pub mod graph;
pub mod optimizer;
pub mod tensor;
pub mod tokenizer;
