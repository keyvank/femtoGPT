use femto_gpt::graph::GraphError;

#[cfg(not(feature = "gpu"))]
fn main() -> Result<(), GraphError> {
    use femto_gpt::gpt::{TrainingState, GPT};
    use femto_gpt::graph::CpuGraph;
    use femto_gpt::optimizer::AdamW;
    use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
    use std::fs;
    use std::io::prelude::*;
    use std::path::Path;

    let training_state_path = Path::new("training_state.dat");

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
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0;

    assert_eq!(num_heads * head_size, embedding_degree);

    println!("Vocab-size: {} unique characters", vocab_size);

    let mut gpt = GPT::new(
        &mut rng,
        CpuGraph::new(),
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        dropout,
        AdamW::new(),
    )?;

    println!("Number of parameters: {}", gpt.num_params());

    // Load training data from train_data directory (If exists)
    // If you want to reuse training_data of a smaller model in a bigger model, you may
    // first start again with a new optimizer by setting load_optimizer=false
    // WARN: YOU CAN ONLY REUSE THE WEIGHTS OF A MODEL WITH DIFFERENT NUM-LAYERS!
    // IT'S NOT POSSIBLE TO CHANGE OTHER PROPERTIES ONCE THE MODEL IS TRAINED!
    if training_state_path.is_file() {
        let mut ts_file = fs::File::open(training_state_path).unwrap();
        let mut bytes = Vec::new();
        ts_file.read_to_end(&mut bytes).unwrap();
        let ts: TrainingState<AdamW> = bincode::deserialize(&bytes).unwrap();
        gpt.set_training_state(ts, true)?;
    }

    println!();
    println!("Starting the training loop... (This make take hours to converge! be patient!)");
    println!();

    let base_lr = 0.001;
    let min_lr = 0.00001;
    let warmup_steps = 100;
    let decay_steps = 50000;

    // Training loop!
    gpt.train(
        &dataset,
        100000,
        batch_size,
        None, // or Some(n), limit backward process to last n computations
        |step| {
            if step < warmup_steps {
                (base_lr / warmup_steps as f32) * step as f32
            } else {
                // Fancy LR tuning, thanks to https://github.com/cutoken!
                f32::max(
                    min_lr,
                    base_lr
                        - (base_lr - min_lr) * (step - warmup_steps) as f32 / decay_steps as f32,
                )
            }
        },
        |gpt| {
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
            let ts = gpt.get_training_state().unwrap();
            let bytes = bincode::serialize(&ts).unwrap();
            fs::write(training_state_path, &bytes).expect("Unable to write file");

            Ok(())
        },
    )?;

    Ok(())
}

#[cfg(feature = "gpu")]
use femto_gpt::graph::Graph;

#[cfg(feature = "gpu")]
use femto_gpt::tensor::*;

#[cfg(feature = "gpu")]
fn pass_graph<G: Graph>(
    mut g: G,
    a_val: &Tensor<f32>,
    b_val: &Tensor<f32>,
) -> Result<Vec<Tensor<f32>>, GraphError> {
    use femto_gpt::funcs::*;

    let a = g.alloc(Tensor::zeros(&[10, 2, 3]), "".into())?;
    let b = g.alloc(Tensor::zeros(&[3, 4]), "".into())?;
    let c = g.call(MatMul::new(), &[a, b])?;
    let d = g.call(Softmax::new(), &[c])?;

    g.load(a, a_val)?;
    g.load(b, b_val)?;
    g.forward(false)?;
    g.zero_grad()?;
    g.backward_all(
        d,
        CrossEntropy::new(4, Tensor::<usize>::zeros(&[10, 2])),
        Some(10),
    )?;

    let ids = vec![a, b, c, d];
    let mut vals = Vec::new();
    for id in ids {
        g.fetch(id, true)?;
        vals.push(g.get_grad(id)?.clone());
    }

    Ok(vals)
}

#[cfg(feature = "gpu")]
fn main() -> Result<(), GraphError> {
    use femto_gpt::graph::gpu;
    use femto_gpt::graph::CpuGraph;

    let mut rng = rand::thread_rng();
    let a_val = Tensor::<f32>::rand(&mut rng, &[10, 2, 3]);
    let b_val = Tensor::<f32>::rand(&mut rng, &[3, 4]);
    let gpu_graph = gpu::GpuGraph::new()?;
    let gpu_result = pass_graph(gpu_graph, &a_val, &b_val)?;

    let cpu_graph = CpuGraph::new();
    let cpu_result = pass_graph(cpu_graph, &a_val, &b_val)?;

    for (t1, t2) in gpu_result.into_iter().zip(cpu_result.into_iter()) {
        let diff = (&t1 - &t2)?;
        let diff_zero = diff.blob().iter().map(|f| *f < 0.000001).all(|t| t);
        println!("{:?}", diff_zero);
    }

    Ok(())
}
