use femto_gpt::gpt::GPT;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;

fn main() {
    let mut rng = rand::thread_rng();

    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    let mut chars = dataset_char
        .chars()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    chars.sort();
    let int_to_ch = chars
        .iter()
        .enumerate()
        .map(|(i, ch)| (i as u32, *ch))
        .collect::<HashMap<u32, char>>();
    let ch_to_int = chars
        .iter()
        .enumerate()
        .map(|(i, ch)| (*ch, i as u32))
        .collect::<HashMap<char, u32>>();
    let dataset = dataset_char
        .chars()
        .map(|ch| ch_to_int.get(&ch).unwrap().clone())
        .collect::<Vec<_>>();

    let batch_size = 10;
    let num_tokens = 64;
    let vocab_size = chars.len();
    let embedding_degree = 64;
    let num_layers = 6;
    let num_heads = 8;
    let head_size = 8;

    let mut gpt = GPT::new(
        &mut rng,
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
    );

    //gpt.load();

    /*gpt.infer(30, |ch| {
        print!("{}", int_to_ch.get(&ch).unwrap());
        std::io::stdout().flush().unwrap();
    });*/

    println!();

    gpt.train(&dataset, 100, batch_size);

    /*println!(
        "Params: {}",
        params.iter().map(|p| g.get(*p).size()).sum::<usize>()
    );*/
}
