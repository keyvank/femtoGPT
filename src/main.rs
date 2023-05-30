mod funcs;
mod graph;
mod optimizer;
mod tensor;

use funcs::*;
use graph::{Graph, TensorId};
use optimizer::AdamW;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::*;
use std::io::prelude::*;
use tensor::{Tensor, TensorMutOps, TensorOps};

fn sample_dataset<R: Rng>(
    dataset: &[u32],
    batch_size: usize,
    context_size: usize,
    rng: &mut R,
) -> (Tensor<u32>, Tensor<u32>) {
    let mut xs: Vec<u32> = Vec::new();
    let mut ys: Vec<u32> = Vec::new();
    for _i in 0..batch_size {
        let start: usize = rng.gen_range(0..dataset.len());
        let all = dataset
            .iter()
            .cycle()
            .skip(start)
            .take(context_size + 1)
            .cloned()
            .collect::<Vec<_>>();
        xs.extend(&all[0..context_size]);
        ys.extend(&all[1..context_size + 1]);
    }

    (
        Tensor::raw(&[batch_size, context_size], xs),
        Tensor::raw(&[batch_size, context_size], ys),
    )
}

fn embed(s: &Tensor<u32>, embedding: &Tensor<f32>) -> Tensor<f32> {
    s.map(0, |s| embedding.get(s.scalar() as usize).into())
}

fn unembed(s: &Tensor<u32>, s_result: &Tensor<f32>, embedding: &mut Tensor<f32>) -> Tensor<f32> {
    let _degree = s_result.shape()[s_result.dim() - 1];
    for (ch, embed) in s.blob().iter().zip(s_result.keep_right(1).inners().iter()) {
        let mut t = embedding.get_mut(*ch as usize);
        t.set(embed.clone());
    }
    Tensor::scalar(0.)
}

fn main() {
    let mut rng = rand::thread_rng();

    let mut g = Graph::new();
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

    let num_attentions = 6;
    let num_heads = 8;
    let head_size = 8;
    let head_size_sqrt_inv = 0.125;

    let mut embedding = Tensor::<f32>::rand(&mut rng, &[vocab_size, embedding_degree]);
    let mut pos_embedding = Tensor::<f32>::rand(&mut rng, &[num_tokens, embedding_degree]);

    let char_inp = g.alloc_input(&[num_tokens, embedding_degree]);
    let pos_inp = g.alloc_input(&[num_tokens, embedding_degree]);
    let inp = g.call(Add::new(), &[char_inp, pos_inp]);

    let mut params: Vec<TensorId> = Vec::new();

    params.extend(&[char_inp, pos_inp]);

    let mut curr_inp = inp;
    for _ in 0..num_attentions {
        let norm_coeff = g.alloc_param(&mut rng, &[embedding_degree]);
        let norm_bias = g.alloc_param(&mut rng, &[embedding_degree]);
        params.extend(&[norm_coeff, norm_bias]);
        let norm_inp = g.call(LayerNorm::new(1), &[curr_inp, norm_coeff, norm_bias]);
        let mut heads = Vec::new();
        for _i in 0..num_heads {
            let k_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            let q_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            let v_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            params.extend(&[k_params, q_params, v_params]);
            let k = g.call(MatMul::new(), &[norm_inp, k_params]);
            let q = g.call(MatMul::new(), &[norm_inp, q_params]);
            let v = g.call(MatMul::new(), &[norm_inp, v_params]);
            let q_t = g.call(Transpose::new(), &[q]);
            let kq = g.call(MatMul::new(), &[k, q_t]);
            let kq_coeff = g.call(Coeff::new(head_size_sqrt_inv), &[kq]);

            let masked_kq = g.call(
                Mask::new(!&Tensor::<bool>::tril(num_tokens), f32::NEG_INFINITY),
                &[kq_coeff],
            );
            let soft_masked_kq = g.call(Softmax::new(), &[masked_kq]);
            let dropped_soft_masked_kq = g.call(Dropout::new(0.2), &[soft_masked_kq]);
            let atten = g.call(MatMul::new(), &[dropped_soft_masked_kq, v]);
            heads.push(atten);
        }
        let cat = g.call(Cat::new(), &heads);

        let proj_params = g.alloc_param(&mut rng, &[num_heads * head_size, embedding_degree]);
        let proj_bias_params = g.alloc_param(&mut rng, &[embedding_degree]);
        let proj_cat = g.call(MatMul::new(), &[cat, proj_params]);

        let proj_cat_bias = g.call(Add::new(), &[proj_cat, proj_bias_params]);
        let dropped_proj_cat_bias = g.call(Dropout::new(0.2), &[proj_cat_bias]);

        let add_atten = g.call(Add::new(), &[norm_inp, dropped_proj_cat_bias]);
        let add_atten_norm_coeff = g.alloc_param(&mut rng, &[embedding_degree]);
        let add_atten_norm_bias = g.alloc_param(&mut rng, &[embedding_degree]);
        let add_atten_norm = g.call(
            LayerNorm::new(1),
            &[add_atten, add_atten_norm_coeff, add_atten_norm_bias],
        );

        let lin1_params = g.alloc_param(&mut rng, &[embedding_degree, 4 * embedding_degree]);
        let bias1_params = g.alloc_param(&mut rng, &[4 * embedding_degree]);
        let lin2_params = g.alloc_param(&mut rng, &[4 * embedding_degree, embedding_degree]);
        let bias2_params = g.alloc_param(&mut rng, &[embedding_degree]);

        let lin1_result = g.call(MatMul::new(), &[add_atten_norm, lin1_params]);
        let lin1_bias_result = g.call(Add::new(), &[lin1_result, bias1_params]);
        let lin1_act = g.call(Relu::new(), &[lin1_bias_result]);
        let lin2_result = g.call(MatMul::new(), &[lin1_act, lin2_params]);
        let lin2_bias_result = g.call(Add::new(), &[lin2_result, bias2_params]);

        params.extend(&[
            proj_params,
            proj_bias_params,
            lin1_params,
            bias1_params,
            lin2_params,
            bias2_params,
            add_atten_norm_coeff,
            add_atten_norm_bias,
        ]);

        curr_inp = g.call(Add::new(), &[add_atten_norm, lin2_bias_result]);
    }

    let norm_out_coeff = g.alloc_param(&mut rng, &[embedding_degree]);
    let norm_out_bias = g.alloc_param(&mut rng, &[embedding_degree]);
    params.extend(&[norm_out_coeff, norm_out_bias]);
    let norm_out = g.call(
        LayerNorm::new(1),
        &[curr_inp, norm_out_coeff, norm_out_bias],
    );
    let to_vocab = g.alloc_param(&mut rng, &[embedding_degree, vocab_size]);
    let to_vocab_bias = g.alloc_param(&mut rng, &[vocab_size]);
    let result_lin = g.call(MatMul::new(), &[norm_out, to_vocab]);
    let result = g.call(Add::new(), &[result_lin, to_vocab_bias]);
    params.extend(&[to_vocab, to_vocab_bias]);

    println!(
        "Params: {}",
        params.iter().map(|p| g.get(*p).size()).sum::<usize>()
    );

    {
        for p in params.iter() {
            if *p != char_inp || *p != pos_inp {
                let mut tensor_file = File::open(format!("tensor_{}.dat", p)).unwrap();
                let mut bytes = Vec::new();
                tensor_file.read_to_end(&mut bytes).unwrap();
                let t: Tensor<f32> = bincode::deserialize(&bytes).unwrap();
                g.load(*p, &t);
            }
        }
        let mut embed_data = File::open("embedding.dat").unwrap();
        let mut bytes = Vec::new();
        embed_data.read_to_end(&mut bytes).unwrap();
        embedding = bincode::deserialize(&bytes).unwrap();

        let mut pos_embed_data = File::open("pos_embedding.dat").unwrap();
        let mut bytes = Vec::new();
        pos_embed_data.read_to_end(&mut bytes).unwrap();
        pos_embedding = bincode::deserialize(&bytes).unwrap();
    }

    let mut opt = AdamW::new(0.00001);
    loop {
        let poses = Tensor::raw(
            &[batch_size, num_tokens],
            (0..num_tokens as u32)
                .cycle()
                .take(num_tokens * batch_size)
                .collect(),
        );
        let (xs, ys) = sample_dataset(&dataset, batch_size, num_tokens, &mut rng);
        g.load(char_inp, &embed(&xs, &embedding));
        g.load(pos_inp, &embed(&poses, &pos_embedding));
        g.forward(true);
        g.zero_grad();
        let err = g.backward_all(result, CrossEntropy::new(vocab_size as u32, ys.clone()));
        println!("Loss: {}", err.mean());
        g.optimize(&mut opt, &params.iter().cloned().collect());
        unembed(&xs, g.get(char_inp), &mut embedding);
        unembed(&poses, g.get(pos_inp), &mut pos_embedding);
        {
            for p in params.iter() {
                if *p != char_inp || *p != pos_inp {
                    let data = bincode::serialize(g.get(*p)).unwrap();
                    fs::write(format!("tensor_{}.dat", p), &data).expect("Unable to write file");
                }
            }
            let embed_data = bincode::serialize(&embedding).unwrap();
            fs::write("embedding.dat", &embed_data).expect("Unable to write file");
            let pos_embed_data = bincode::serialize(&pos_embedding).unwrap();
            fs::write("pos_embedding.dat", &pos_embed_data).expect("Unable to write file");
        }

        /*{
            let mut cnt = 1;
            let mut context = vec![0; num_tokens];
            for _ in 0..256 {
                g.load(
                    inp,
                    &embed(&Tensor::raw(&[1, num_tokens], context.clone()), &embedding),
                );
                g.forward(false);
                let next_ch = g.get(result).argmax().blob()[cnt - 1];
                println!(
                    "{}",
                    &context
                        .iter()
                        .map(|i| int_to_ch.get(i).unwrap())
                        .collect::<String>()[..cnt]
                );
                if cnt == 64 {
                    context.remove(0);
                    context.push(0);
                    cnt -= 1;
                }
                context[cnt] = next_ch;
                cnt += 1;
            }
            println!();
        }*/
    }
}
