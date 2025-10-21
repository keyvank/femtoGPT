use femto_gpt::gpt::{TrainingState, GPT};
use femto_gpt::graph::GraphError;
use femto_gpt::optimizer::AdamW;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
use std::fs;
use std::io::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;

/// Command-line interface definition for the model training and inference application.
///
/// This enum defines two operational modes — [`Train`](Cli::Train) and [`Infer`](Cli::Infer) —
/// that control how the program behaves.
///
/// (Note: Add `--features gpu` in order to leverage GPU speedups!)
/// If you want to train using a GPU, you will first need to make sure your GPU drivers are correctly installed on your system, and their OpenCL runtimes are available.
#[derive(StructOpt, Debug)]
enum Cli {
    /// **Training mode** — used to train or fine-tune a model on a given dataset.
    ///
    /// This mode reads a text dataset and updates or creates a serialized model state file.
    /// Example:
    /// ```bash
    /// cargo run --release train --dataset dataset.txt --model training_state.dat
    /// ```
    Train {
        #[structopt(long, default_value = "dataset.txt")]
        dataset: PathBuf,
        #[structopt(long, default_value = "training_state.dat")]
        model: PathBuf,
    },

    /// **Inference mode** — used to generate predictions or text completions from a trained model.
    ///
    /// This mode loads a trained model and a tokenizer dataset, processes a given text prompt,
    /// and outputs generated text samples.
    ///
    /// Example:
    /// ```bash
    /// cargo run --release infer --tokenizer-dataset dataset.txt --model training_state.dat \
    ///     --prompt "Once upon a time" --count 5 --temperature 0.7
    /// ```
    Infer {
        #[structopt(long, default_value = "dataset.txt")]
        tokenizer_dataset: PathBuf,
        #[structopt(long, default_value = "training_state.dat")]
        model: PathBuf,
        #[structopt(long)]
        prompt: String,
        #[structopt(long, default_value = "100")]
        count: usize,
        #[structopt(long, default_value = "0.5")]
        temperature: f32,
    },
}

/// Entry point for femtoGPT
///
/// Depending on whether the `gpu` feature is enabled at compile time, the
/// computation graph is constructed using either a CPU or GPU backend.
///
/// # Returns
/// Returns `Ok(())` on success, or a [`GraphError`] if model initialization or computation fails.
///
fn main() -> Result<(), GraphError> {
    // Choose CPU or GPU computation graph depending on feature flags.
    #[cfg(not(feature = "gpu"))]
    let graph = femto_gpt::graph::CpuGraph::new();
    #[cfg(not(feature = "gpu"))]
    let is_gpu = false;

    #[cfg(feature = "gpu")]
    let graph = femto_gpt::graph::gpu::GpuGraph::new()?; // TODO: may fail if GPU not available, gracefully handle
    #[cfg(feature = "gpu")]
    let is_gpu = true;

    // Model hyperparameters
    let batch_size = 32; // Number of samples processed in one training step.
    let num_tokens = 64; // Maximum number of tokens per sequence.
    let embedding_degree = 64; // Dimensionality of embedding vectors.
    let num_layers = 4; // Number of transformer layers.
    let num_heads = 4; // Number of attention heads.
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0; // Dropout probability (disabled by default).
    assert_eq!(num_heads * head_size, embedding_degree);

    // Parse CLI arguments to determine mode (Train / Infer)
    let cli = Cli::from_args();

    // CLI Mode: Inference
    match cli {
        Cli::Infer {
            tokenizer_dataset,
            model,
            prompt,
            count,
            temperature,
        } => {
            let training_state_path = &model.clone();

            let mut rng = rand::thread_rng();

            // Create a unique char-to-int mapping for all unique characters inside our dataset
            //
            // The dataset is treated as a flat text corpus from which all unique characters
            // will be extracted to define the vocabulary of the model.
            let dataset_char = fs::read_to_string(tokenizer_dataset).expect(
                "Failed to read tokenizer dataset file. Ensure the path exists and is readable.",
            );
            let tokenizer = SimpleTokenizer::new(&dataset_char);

            assert_eq!(num_heads * head_size, embedding_degree); // runtime sanity check to ensure transformer hyperparameters are consistent

            // number of unique characters in the tokenizer vocabulary.
            let vocab_size = tokenizer.vocab_size();
            println!("Vocab-size: {} unique characters", vocab_size);
            
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;

            gpt.sync()?;

            let mut ts_file = fs::File::open(&training_state_path).unwrap();
            let mut bytes = Vec::new();
            ts_file.read_to_end(&mut bytes).unwrap();
            let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
            gpt.set_training_state(ts, true)?;

            println!("Generating text:");

            let inference = gpt.infer(
                &mut rng,
                &tokenizer.tokenize(&prompt),
                count,
                temperature,
                |_ch| {},
            )?;

            // Generate 100 character with the currently trained model
            println!("{}", tokenizer.untokenize(&inference));

            Ok(())
        }
        Cli::Train { dataset, model } => {
            let training_state_path = &model.clone();

            let mut rng = rand::thread_rng();

            // Create a unique char-to-int mapping for all unique characters inside our dataset
            let dataset_char =
                fs::read_to_string(dataset).expect("Should have been able to read the file");
            let tokenizer = SimpleTokenizer::new(&dataset_char);

            let dataset = tokenizer.tokenize(&dataset_char);

            let vocab_size = tokenizer.vocab_size();
            println!("Vocab-size: {} unique characters", vocab_size);
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;

            gpt.sync()?;

            println!("Number of parameters: {}", gpt.num_params());

            // Load training data from train_data directory (If exists)
            // If you want to reuse training_data of a smaller model in a bigger model, you may
            // first start again with a new optimizer by setting load_optimizer=false
            // WARN: YOU CAN ONLY REUSE THE WEIGHTS OF A MODEL WITH DIFFERENT NUM-LAYERS!
            // IT'S NOT POSSIBLE TO CHANGE OTHER PROPERTIES ONCE THE MODEL IS TRAINED!
            if training_state_path.is_file() {
                let mut ts_file = fs::File::open(&training_state_path).unwrap();
                let mut bytes = Vec::new();
                ts_file.read_to_end(&mut bytes).unwrap();
                let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
                gpt.set_training_state(ts, true)?;
            }

            println!();
            println!(
                "Starting the training loop... (This make take hours to converge! be patient!)"
            );
            println!();

            let base_lr = 0.001;
            let min_lr = 0.00001;
            let warmup_steps = 100;
            let decay_steps = 50000;

            let learning_rate = |step| {
                if step < warmup_steps {
                    (base_lr / warmup_steps as f32) * step as f32
                } else {
                    // Fancy LR tuning, thanks to https://github.com/cutoken!
                    f32::max(
                        min_lr,
                        base_lr
                            - (base_lr - min_lr) * (step - warmup_steps) as f32
                                / decay_steps as f32,
                    )
                }
            };

            let callback = |gpt: &mut GPT<_>| {
                let mut rng = rand::thread_rng();
                let inference_temperature = 0.5; // How creative? 0.0 min 1.0 max

                println!("Generating text:");

                let inference = gpt.infer(
                    &mut rng,
                    &tokenizer.tokenize("\n"),
                    100,
                    inference_temperature,
                    |_ch| {},
                )?;

                // Generate 100 character with the currently trained model before
                // starting the training loop.
                println!("{}", tokenizer.untokenize(&inference));

                println!("Saving the model...");
                gpt.sync().unwrap();
                let ts = gpt.get_training_state().unwrap();
                let bytes = bincode::serialize(&ts).unwrap();
                fs::write(training_state_path, &bytes).expect("Unable to write file");

                Ok(())
            };

            // Training loop!
            #[cfg(not(feature = "gpu"))]
            gpt.train_cpu(
                &dataset,
                100000,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            #[cfg(feature = "gpu")]
            gpt.train(
                &dataset,
                100000,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            Ok(())
        }
    }
}
