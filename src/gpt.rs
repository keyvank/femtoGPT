use crate::funcs::*;
use crate::graph::{Graph, TensorId};
use crate::optimizer::Optimizer;
use crate::tensor::{Tensor, TensorMutOps, TensorOps};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

use std::fs;
use std::fs::*;
use std::io::prelude::*;

pub struct GPT<O: Optimizer, R: Rng> {
    rng: R,
    graph: Graph,
    vocab_size: usize,
    num_tokens: usize,
    params: Vec<TensorId>,
    token_embedding: TensorId,
    pos_embedding: TensorId,
    token_input: TensorId,
    pos_input: TensorId,
    output: TensorId,
    optimizer: O,
}

fn sample_dataset<R: Rng>(
    dataset: &[usize],
    batch_size: usize,
    context_size: usize,
    rng: &mut R,
) -> (Tensor<usize>, Tensor<usize>) {
    let mut xs: Vec<usize> = Vec::with_capacity(batch_size * context_size);
    let mut ys: Vec<usize> = Vec::with_capacity(batch_size * context_size);
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

fn embed(s: &Tensor<usize>, embedding: &Tensor<f32>) -> Tensor<f32> {
    s.map(0, |s| embedding.get(s.scalar()).into())
}

use std::collections::HashMap;
fn unembed(s: &Tensor<usize>, s_result: &Tensor<f32>, embedding: &mut Tensor<f32>) {
    let mut embeds: HashMap<usize, Vec<Tensor<f32>>> = HashMap::new();
    for (ch, embed) in s.blob().iter().zip(s_result.keep_right(1).inners().iter()) {
        embeds.entry(*ch).or_default().push(embed.clone().into());
    }
    for (ch, vals) in embeds {
        let mut avg = Tensor::scalar(0.);
        for v in vals.iter() {
            avg = &avg + v;
        }
        avg = &avg * &Tensor::scalar(1. / vals.len() as f32);
        let mut t = embedding.get_mut(ch);
        t.set(avg.clone());
    }
}

fn select<T: TensorOps<f32>>(t: &T) -> usize {
    let t = Softmax::new().run(&[&Tensor::<f32>::raw(t.shape(), t.blob().to_vec())], false);
    let mut rng = rand::thread_rng();
    let mut ts = t.blob().iter().cloned().enumerate().collect::<Vec<_>>();
    ts.sort_by_key(|(_, b)| (b * 1000.) as usize);
    let dice = rng.gen_range(0f32..1f32);
    let mut accum = 0.;
    for (id, t) in ts.iter().rev() {
        accum += t;
        if dice < accum {
            return *id;
        }
    }
    panic!();
}

impl<O: Optimizer, R: Rng> GPT<O, R> {
    pub fn new(
        mut rng: R,
        vocab_size: usize,
        embedding_degree: usize,
        num_tokens: usize,
        num_layers: usize,
        num_heads: usize,
        head_size: usize,
        dropout: f32,
        optimizer: O,
    ) -> Self {
        let mut g = Graph::new();

        let token_embedding = g.alloc_rand(&mut rng, &[vocab_size, embedding_degree]);
        let pos_embedding = g.alloc_rand(&mut rng, &[num_tokens, embedding_degree]);

        let token_input = g.alloc_rand(&mut rng, &[1, num_tokens, embedding_degree]);
        let pos_input = g.alloc_rand(&mut rng, &[1, num_tokens, embedding_degree]);
        let inp = g.call(Add::new(), &[token_input, pos_input]);

        // Keep track of tensor-ids of learnable tensors!
        let mut params: Vec<TensorId> = Vec::new();

        params.extend(&[token_embedding, pos_embedding]);

        let mut curr_inp = inp;
        for _ in 0..num_layers {
            // Normalize input before applying multi-head attention
            let norm_coeff = g.alloc_rand(&mut rng, &[embedding_degree]);
            let norm_bias = g.alloc_rand(&mut rng, &[embedding_degree]);
            params.extend(&[norm_coeff, norm_bias]);
            let norm_inp = g.call(LayerNorm::new(), &[curr_inp, norm_coeff, norm_bias]);

            let mut heads = Vec::new();

            // Multi-head Attention
            for _ in 0..num_heads {
                let k_params = g.alloc_rand(&mut rng, &[embedding_degree, head_size]);
                let q_params = g.alloc_rand(&mut rng, &[embedding_degree, head_size]);
                let v_params = g.alloc_rand(&mut rng, &[embedding_degree, head_size]);
                params.extend(&[k_params, q_params, v_params]);
                let k = g.call(MatMul::new(), &[norm_inp, k_params]);
                let q = g.call(MatMul::new(), &[norm_inp, q_params]);
                let v = g.call(MatMul::new(), &[norm_inp, v_params]);
                let q_t = g.call(Transpose::new(), &[q]);
                let kq = g.call(MatMul::new(), &[k, q_t]);

                let head_size_sqrt_inv = (head_size as f32).powf(-0.5);
                let kq_coeff = g.call(Coeff::new(head_size_sqrt_inv), &[kq]);

                let masked_kq = g.call(
                    Mask::new(!&Tensor::<bool>::tril(num_tokens), f32::NEG_INFINITY),
                    &[kq_coeff],
                );
                let soft_masked_kq = g.call(Softmax::new(), &[masked_kq]);
                let dropped_soft_masked_kq = g.call(Dropout::new(dropout), &[soft_masked_kq]);
                let atten = g.call(MatMul::new(), &[dropped_soft_masked_kq, v]);
                heads.push(atten);
            }

            // Concat head results and project into embedding_degree
            let cat = g.call(Cat::new(), &heads);
            let proj_params = g.alloc_rand(&mut rng, &[num_heads * head_size, embedding_degree]);
            let proj_bias_params = g.alloc_rand(&mut rng, &[embedding_degree]);
            let proj_cat = g.call(MatMul::new(), &[cat, proj_params]);
            let proj_cat_bias = g.call(Add::new(), &[proj_cat, proj_bias_params]);
            let dropped_proj_cat_bias = g.call(Dropout::new(dropout), &[proj_cat_bias]);

            // Add attention results to input and then normalize
            let add_atten = g.call(Add::new(), &[norm_inp, dropped_proj_cat_bias]);
            let add_atten_norm_coeff = g.alloc_rand(&mut rng, &[embedding_degree]);
            let add_atten_norm_bias = g.alloc_rand(&mut rng, &[embedding_degree]);
            let add_atten_norm = g.call(
                LayerNorm::new(),
                &[add_atten, add_atten_norm_coeff, add_atten_norm_bias],
            );

            // A feed-forward layer:
            // Linear embedding_degree -> 4*embedding_degree
            // Relu
            // Linear 4*embedding_degree -> embedding_degree
            let lin1_params = g.alloc_rand(&mut rng, &[embedding_degree, 4 * embedding_degree]);
            let bias1_params = g.alloc_rand(&mut rng, &[4 * embedding_degree]);
            let lin2_params = g.alloc_rand(&mut rng, &[4 * embedding_degree, embedding_degree]);
            let bias2_params = g.alloc_rand(&mut rng, &[embedding_degree]);
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

        // Normalize the output after the last layer
        let norm_out_coeff = g.alloc_rand(&mut rng, &[embedding_degree]);
        let norm_out_bias = g.alloc_rand(&mut rng, &[embedding_degree]);
        params.extend(&[norm_out_coeff, norm_out_bias]);
        let norm_out = g.call(LayerNorm::new(), &[curr_inp, norm_out_coeff, norm_out_bias]);

        // Map from embedding_degree to vocab_size through a linear layer
        let to_vocab = g.alloc_rand(&mut rng, &[embedding_degree, vocab_size]);
        let to_vocab_bias = g.alloc_rand(&mut rng, &[vocab_size]);
        let result_lin = g.call(MatMul::new(), &[norm_out, to_vocab]);
        let output = g.call(Add::new(), &[result_lin, to_vocab_bias]);
        params.extend(&[to_vocab, to_vocab_bias]);

        Self {
            rng,
            graph: g,
            vocab_size,
            num_tokens,
            params,
            token_input,
            pos_input,
            output,
            token_embedding,
            pos_embedding,
            optimizer,
        }
    }

    pub fn num_params(&self) -> usize {
        self.params
            .iter()
            .map(|p| self.graph.get(*p).size())
            .sum::<usize>()
    }

    pub fn load(&mut self) {
        if std::path::Path::new("train_data").is_dir() {
            for p in self.params.iter() {
                let mut tensor_file = File::open(format!("train_data/tensor_{}.dat", p)).unwrap();
                let mut bytes = Vec::new();
                tensor_file.read_to_end(&mut bytes).unwrap();
                let t: Tensor<f32> = bincode::deserialize(&bytes).unwrap();
                self.graph.load(*p, &t);
            }

            let mut opt_data = File::open("train_data/optimizer.dat").unwrap();
            let mut bytes = Vec::new();
            opt_data.read_to_end(&mut bytes).unwrap();
            self.optimizer = bincode::deserialize(&bytes).unwrap();
        }
    }

    pub fn save(&self) {
        fs::create_dir_all("train_data").unwrap();
        for p in self.params.iter() {
            let data = bincode::serialize(self.graph.get(*p)).unwrap();
            fs::write(format!("train_data/tensor_{}.dat", p), &data).expect("Unable to write file");
        }
        let opt_data = bincode::serialize(&self.optimizer).unwrap();
        fs::write("train_data/optimizer.dat", &opt_data).expect("Unable to write file");
    }

    pub fn train(&mut self, dataset: &[usize], num_batches: usize, batch_size: usize) {
        for i in 0..num_batches {
            let timer = Instant::now();
            let (graphs, errs): (Vec<Graph>, Vec<f32>) = (0..batch_size)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let mut graph = self.graph.clone();
                    let poses = Tensor::raw(
                        &[1, self.num_tokens],
                        (0..self.num_tokens)
                            .cycle()
                            .take(self.num_tokens * 1)
                            .collect(),
                    );
                    let (xs, ys) = sample_dataset(dataset, 1, self.num_tokens, &mut rng);
                    graph.load(
                        self.token_input,
                        &embed(&xs, graph.get(self.token_embedding)),
                    );
                    graph.load(
                        self.pos_input,
                        &embed(&poses, graph.get(self.pos_embedding)),
                    );
                    graph.forward(true);
                    graph.zero_grad();
                    let err = graph
                        .backward_all(self.output, CrossEntropy::new(self.vocab_size, ys.clone()));
                    let mut token_embedding_grad =
                        Tensor::<f32>::zeros(graph.get(self.token_embedding).shape());
                    let mut pos_embedding_grad =
                        Tensor::<f32>::zeros(graph.get(self.pos_embedding).shape());
                    unembed(
                        &xs,
                        graph.get_grad(self.token_input),
                        &mut token_embedding_grad,
                    );
                    unembed(
                        &poses,
                        graph.get_grad(self.pos_input),
                        &mut pos_embedding_grad,
                    );
                    graph.load_grad(self.token_embedding, &token_embedding_grad);
                    graph.load_grad(self.pos_embedding, &pos_embedding_grad);
                    (graph, err)
                })
                .unzip();
            for (id, avg) in self.params.iter().map(|id| {
                let mut avg = Tensor::<f32>::scalar(0.);
                graphs.iter().for_each(|g| avg = &avg + g.get_grad(*id));
                avg = avg.map_values(|f| f / graphs.len() as f32);
                (id, avg)
            }) {
                self.graph.load_grad(*id, &avg);
            }
            let avg_loss = errs.iter().sum::<f32>() / errs.len() as f32;
            println!(
                "Step: {} Loss: {} (Elapsed: {}ms)",
                i,
                avg_loss,
                timer.elapsed().as_millis()
            );
            self.graph
                .optimize(&mut self.optimizer, &self.params.iter().cloned().collect());
            if i % 50 == 0 {
                println!("Saving the model...");
                self.save();
            }
        }
    }

    pub fn infer<F: Fn(usize) -> ()>(&mut self, prompt: &[usize], count: usize, callback: F) {
        let mut cnt = prompt.len();
        let mut context = vec![0; self.num_tokens];
        context[..prompt.len()].copy_from_slice(prompt);
        let poses = Tensor::raw(&[1, self.num_tokens], (0..self.num_tokens).collect());
        self.graph.load(
            self.pos_input,
            &embed(&poses, &self.graph.get(self.pos_embedding)),
        );
        for _ in 0..count {
            self.graph.load(
                self.token_input,
                &embed(
                    &Tensor::raw(&[1, self.num_tokens], context.clone()),
                    self.graph.get(self.token_embedding),
                ),
            );
            self.graph.forward(false);
            let next_ch = select(&self.graph.get(self.output).get(0).get(cnt - 1));
            callback(next_ch);
            if cnt == self.num_tokens {
                context.remove(0);
                context.push(0);
                cnt -= 1;
            }
            context[cnt] = next_ch;
            cnt += 1;
        }
    }
}
