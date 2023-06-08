use femto_gpt::gpt::GPT;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};

use std::fs;
use std::io::Write;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    // learning rate decay iterations
    #[arg(short, long, default_value_t = 0)]
    lr_decay_iter: usize
}

fn main() {
    let args = Cli::parse();
    let mut rng = rand::thread_rng();

    // Create a unique char-to-int mapping for all unique characters inside our dataset
    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    let tokenizer = SimpleTokenizer::new(&dataset_char);

    let dataset = tokenizer.tokenize(&dataset_char);

    let batch_size = 32;

    let num_tokens = 64;
    let vocab_size = tokenizer.vocab_size();
    let embedding_degree = 64;
    let num_layers = 6;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0;

    assert_eq!(num_heads * head_size, embedding_degree);

    println!("Vocab-size: {} unique characters", vocab_size);

    let mut gpt = GPT::new(
        &mut rng,
        args.lr_decay_iter,
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        dropout,
        femto_gpt::optimizer::AdamW::new(0.001),
    );

    println!("Number of parameters: {}", gpt.num_params());

    // Load training data from train_data directory (If exists)
    gpt.load();

    println!("Generating text:");

    // Generate 100 character with the currently trained model before
    // starting the training loop.
    gpt.infer(&tokenizer.tokenize("\n"), 100, |ch| {
        print!("{}", tokenizer.untokenize(&[ch]));
        std::io::stdout().flush().unwrap();
    });

    println!();
    println!("Starting the training loop... (This make take hours to converge! be patient!)");
    println!();

    // Training loop!
    gpt.train(&dataset, 100000, batch_size);
}
