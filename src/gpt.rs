use crate::funcs::*;
use crate::graph::{Graph, GraphError, TensorId};
use crate::optimizer::Optimizer;
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorMutOps, TensorOps};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState<O: Clone> {
    pub tensors: HashMap<String, Tensor<f32>>,
    pub optimizer: O,
}

pub struct GPT<G: Graph, O: Optimizer> {
    graph: G,
    vocab_size: usize,
    num_tokens: usize,
    params: Vec<TensorId>,
    token_embedding: TensorId,
    token_input: TensorId,
    pos_input: TensorId,
    output: TensorId,
    optimizer: O,
    pos_input_fixed: Tensor<f32>,
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
        Tensor::raw(&[batch_size, context_size], xs).unwrap(),
        Tensor::raw(&[batch_size, context_size], ys).unwrap(),
    )
}

use std::collections::HashMap;
fn unembed(
    s: &Tensor<usize>,
    s_result: &Tensor<f32>,
    embedding: &mut Tensor<f32>,
) -> Result<(), TensorError> {
    let mut embeds: HashMap<usize, Vec<Tensor<f32>>> = HashMap::new();
    for (ch, embed) in s.blob().iter().zip(s_result.keep_right(1)?.inners().iter()) {
        embeds.entry(*ch).or_default().push(embed.clone().into());
    }
    for (ch, vals) in embeds {
        let mut avg = Tensor::scalar(0.);
        for v in vals.iter() {
            avg = (&avg + v)?;
        }
        avg = (&avg * &Tensor::scalar(1. / vals.len() as f32))?;
        embedding.get_mut(ch)?.set(avg.clone())?;
    }
    Ok(())
}

fn select<R: Rng, T: TensorOps<f32>>(
    rng: &mut R,
    t: &T,
    temperature: f32,
) -> Result<usize, TensorError> {
    let t = Softmax::new().run(
        &[&GeneralTensor::Float(Tensor::<f32>::raw(
            t.shape(),
            t.blob().to_vec(),
        )?)],
        false,
    )?;
    let mut ts = t.blob().iter().cloned().enumerate().collect::<Vec<_>>();
    ts.sort_by_key(|(_, b)| (b * 1000.) as usize);
    let dice = rng.gen_range(0.0..temperature);
    let mut accum = 0.;
    for (id, t) in ts.iter().rev() {
        accum += t;
        if dice < accum {
            return Ok(*id);
        }
    }
    panic!();
}

fn pos_encode_inter(num_tokens: usize, embedding_size: usize) -> Tensor<f32> {
    let mut raw_new = Vec::new();
    let cols = embedding_size;
    let rows = num_tokens;
    for row in 0..rows {
        for col in 0..cols {
            let k = row as f32;
            let i = (col / 2) as f32;
            let factor = 10000f32.powf(2f32 * i / embedding_size as f32);

            let pos = if col % 2 == 0 {
                (k / factor).sin()
            } else {
                (k / factor).cos()
            };

            raw_new.push(pos);
        }
    }

    Tensor::raw(&[rows, cols], raw_new).unwrap()
}

impl<G: Graph, O: Optimizer> GPT<G, O> {
    pub fn new<R: Rng>(
        rng: &mut R,
        mut g: G,
        vocab_size: usize,
        embedding_degree: usize,
        num_tokens: usize,
        num_layers: usize,
        num_heads: usize,
        head_size: usize,
        dropout: f32,
        optimizer: O,
    ) -> Result<Self, GraphError> {
        let token_embedding = g.alloc(
            Tensor::<f32>::rand(rng, &[vocab_size, embedding_degree]),
            "token_embedding".into(),
        )?;
        let token_input = g.alloc(
            Tensor::<f32>::rand(rng, &[num_tokens, embedding_degree]),
            "token_input".into(),
        )?;
        let pos_input = g.alloc(
            Tensor::<f32>::rand(rng, &[num_tokens, embedding_degree]),
            "pos_input".into(),
        )?;
        let inp = g.call(Add::new(), &[token_input, pos_input])?;

        // Keep track of tensor-ids of learnable tensors!
        let mut params: Vec<TensorId> = Vec::new();

        params.extend(&[token_embedding]);

        let mut curr_inp = inp;
        for l in 0..num_layers {
            // Normalize input before applying multi-head attention
            let norm_coeff = g.alloc(
                Tensor::<f32>::rand(rng, &[embedding_degree]),
                format!("norm_{}_coeff", l),
            )?;
            let norm_bias = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                format!("norm_{}_bias", l),
            )?;
            params.extend(&[norm_coeff, norm_bias]);
            let norm_inp = g.call(LayerNorm::new(), &[curr_inp, norm_coeff, norm_bias])?;

            let mut heads = Vec::new();

            // Multi-head Attention
            for h in 0..num_heads {
                let k_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    format!("head_{}_{}_k", l, h),
                )?;
                let q_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    format!("head_{}_{}_q", l, h),
                )?;
                let v_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    format!("head_{}_{}_v", l, h),
                )?;
                params.extend(&[k_params, q_params, v_params]);
                let k = g.call(MatMul::new(), &[norm_inp, k_params])?;
                let q = g.call(MatMul::new(), &[norm_inp, q_params])?;
                let v = g.call(MatMul::new(), &[norm_inp, v_params])?;
                let q_t = g.call(Transpose::new(), &[q])?;
                let kq = g.call(MatMul::new(), &[k, q_t])?;

                let head_size_sqrt_inv = (head_size as f32).powf(-0.5);
                let kq_coeff = g.call(Coeff::new(head_size_sqrt_inv), &[kq])?;

                let masked_kq = g.call(
                    Mask::new(!&Tensor::<bool>::tril(num_tokens), f32::NEG_INFINITY),
                    &[kq_coeff],
                )?;
                let soft_masked_kq = g.call(Softmax::new(), &[masked_kq])?;
                let dropped_soft_masked_kq = g.call(Dropout::new(dropout), &[soft_masked_kq])?;
                let atten = g.call(MatMul::new(), &[dropped_soft_masked_kq, v])?;
                heads.push(atten);
            }

            // Concat head results and project into embedding_degree
            let cat = g.call(Cat::new(), &heads)?;
            let proj_params = g.alloc(
                Tensor::<f32>::rand(rng, &[num_heads * head_size, embedding_degree]),
                format!("proj_{}_weights", l),
            )?;
            let proj_bias_params = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                format!("proj_{}_bias", l),
            )?;
            let proj_cat = g.call(MatMul::new(), &[cat, proj_params])?;
            let proj_cat_bias = g.call(Add::new(), &[proj_cat, proj_bias_params])?;
            let dropped_proj_cat_bias = g.call(Dropout::new(dropout), &[proj_cat_bias])?;

            // Add attention results to input and then normalize
            let add_atten = g.call(Add::new(), &[norm_inp, dropped_proj_cat_bias])?;
            let add_atten_norm_coeff = g.alloc(
                Tensor::<f32>::rand(rng, &[embedding_degree]),
                format!("atten_norm_{}_coeff", l),
            )?;
            let add_atten_norm_bias = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                format!("atten_norm_{}_bias", l),
            )?;
            let add_atten_norm = g.call(
                LayerNorm::new(),
                &[add_atten, add_atten_norm_coeff, add_atten_norm_bias],
            )?;

            // A feed-forward layer:
            // Linear embedding_degree -> 4*embedding_degree
            // Relu
            // Linear 4*embedding_degree -> embedding_degree
            let lin1_params = g.alloc(
                Tensor::<f32>::rand(rng, &[embedding_degree, 4 * embedding_degree]),
                format!("feedforward1_{}_weights", l),
            )?;
            let bias1_params = g.alloc(
                Tensor::<f32>::zeros(&[4 * embedding_degree]),
                format!("feedforward1_{}_bias", l),
            )?;
            let lin1_result = g.call(MatMul::new(), &[add_atten_norm, lin1_params])?;
            let lin1_bias_result = g.call(Add::new(), &[lin1_result, bias1_params])?;
            let lin1_act = g.call(Relu::new(), &[lin1_bias_result])?;
            let lin2_params = g.alloc(
                Tensor::<f32>::rand(rng, &[4 * embedding_degree, embedding_degree]),
                format!("feedforward2_{}_weights", l),
            )?;
            let bias2_params = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                format!("feedforward2_{}_bias", l),
            )?;
            let lin2_result = g.call(MatMul::new(), &[lin1_act, lin2_params])?;
            let lin2_bias_result = g.call(Add::new(), &[lin2_result, bias2_params])?;

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

            curr_inp = g.call(Add::new(), &[add_atten_norm, lin2_bias_result])?;
        }

        // Normalize the output after the last layer
        let norm_out_coeff = g.alloc(
            Tensor::<f32>::rand(rng, &[embedding_degree]),
            format!("head_norm_coeff"),
        )?;
        let norm_out_bias = g.alloc(
            Tensor::<f32>::zeros(&[embedding_degree]),
            format!("head_norm_bias"),
        )?;
        params.extend(&[norm_out_coeff, norm_out_bias]);
        let norm_out = g.call(LayerNorm::new(), &[curr_inp, norm_out_coeff, norm_out_bias])?;

        // Map from embedding_degree to vocab_size through a linear layer
        let to_vocab = g.alloc(
            Tensor::<f32>::rand(rng, &[embedding_degree, vocab_size]),
            format!("head_map_weights"),
        )?;
        let to_vocab_bias = g.alloc(
            Tensor::<f32>::zeros(&[vocab_size]),
            format!("head_map_bias"),
        )?;
        let result_lin = g.call(MatMul::new(), &[norm_out, to_vocab])?;
        let output = g.call(Add::new(), &[result_lin, to_vocab_bias])?;
        params.extend(&[to_vocab, to_vocab_bias]);

        Ok(Self {
            graph: g,
            vocab_size,
            num_tokens,
            params,
            token_input,
            pos_input,
            output,
            token_embedding,
            optimizer,
            pos_input_fixed: pos_encode_inter(num_tokens, embedding_degree),
        })
    }

    pub fn num_params(&self) -> usize {
        self.params
            .iter()
            .map(|p| self.graph.get(*p).unwrap().as_float().unwrap().size())
            .sum::<usize>()
    }

    pub fn set_training_state(
        &mut self,
        training_state: TrainingState<O>,
        load_optimizer: bool,
    ) -> Result<(), GraphError> {
        for p in self.params.iter() {
            let name = self.graph.name_of(*p)?;
            if let Some(t) = training_state.tensors.get(name) {
                self.graph.load(*p, t)?;
            }
        }
        if load_optimizer {
            self.optimizer = training_state.optimizer;
        }
        Ok(())
    }

    pub fn get_training_state(&self) -> Result<TrainingState<O>, GraphError> {
        let mut state = TrainingState {
            tensors: Default::default(),
            optimizer: self.optimizer.clone(),
        };
        for p in self.params.iter() {
            let k = self.graph.name_of(*p)?.to_string();
            let v = self.graph.get(*p)?.as_float()?.clone();
            state.tensors.insert(k, v);
        }
        Ok(state)
    }

    pub fn train<F: Fn(usize) -> f32, C: Fn(&Self) -> Result<(), GraphError>>(
        &mut self,
        dataset: &[usize],
        num_batches: usize,
        batch_size: usize,
        limit: Option<usize>,
        learning_rate: F,
        callback: C,
    ) -> Result<(), GraphError>
    where
        G: Clone + Send + Sync,
    {
        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        for i in 0..num_batches {
            let timer = Instant::now();
            let (graphs, errs): (Vec<G>, Vec<f32>) = (0..batch_size)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::thread_rng();
                    let mut graph = self.graph.clone();
                    let (xs, ys) = sample_dataset(dataset, 1, self.num_tokens, &mut rng);
                    graph.embed(self.token_input, self.token_embedding, &xs)?;

                    graph.forward(true)?;
                    graph.zero_grad();
                    let err = graph.backward_all(
                        self.output,
                        CrossEntropy::new(self.vocab_size, ys.clone()),
                        limit,
                    )?;
                    let mut token_embedding_grad =
                        Tensor::<f32>::zeros(graph.get(self.token_embedding)?.as_float()?.shape());
                    unembed(
                        &xs,
                        graph.get_grad(self.token_input)?,
                        &mut token_embedding_grad,
                    )?;
                    graph.load_grad(self.token_embedding, &token_embedding_grad)?;
                    Ok((graph, err))
                })
                .collect::<Result<Vec<(G, f32)>, GraphError>>()?
                .into_iter()
                .unzip();
            for (id, avg) in self
                .params
                .par_iter()
                .map(|id| {
                    let mut avg = Tensor::<f32>::scalar(0.);
                    for g in graphs.iter() {
                        avg = (&avg + g.get_grad(*id)?)?;
                    }
                    avg = avg.map_values(|f| f / graphs.len() as f32);
                    Ok((id, avg))
                })
                .collect::<Result<Vec<_>, GraphError>>()?
            {
                self.graph.load_grad(*id, &avg)?;
            }
            let avg_loss = errs.iter().sum::<f32>() / errs.len() as f32;
            let lr = learning_rate(self.optimizer.step_num());
            self.graph.optimize(
                &mut self.optimizer,
                &self.params.iter().cloned().collect(),
                lr,
            )?;
            if i % 50 == 0 {
                callback(self)?;
            }
            println!(
                "Step: {} Loss: {} (Elapsed: {}ms)",
                self.optimizer.step_num(),
                avg_loss,
                timer.elapsed().as_millis()
            );
        }
        Ok(())
    }

    pub fn infer<R: Rng, F: Fn(usize) -> ()>(
        &self,
        rng: &mut R,
        prompt: &[usize],
        count: usize,
        temperature: f32,
        callback: F,
    ) -> Result<Vec<usize>, GraphError>
    where
        G: Clone + Send + Sync,
    {
        let mut cnt = prompt.len();
        let mut context = vec![0; self.num_tokens];
        context[..prompt.len()].copy_from_slice(prompt);
        let mut graph = self.graph.clone();

        graph.load(self.pos_input, &self.pos_input_fixed)?;

        for ch in prompt {
            callback(*ch);
        }
        let mut chs = prompt.to_vec();
        for _ in 0..count {
            graph.embed(
                self.token_input,
                self.token_embedding,
                &Tensor::raw(&[self.num_tokens], context.clone())?,
            )?;
            graph.forward(false)?;
            let next_ch = select(
                rng,
                &graph.get(self.output)?.as_float()?.get(cnt - 1)?,
                temperature,
            )?;
            chs.push(next_ch);
            callback(next_ch);
            if cnt == self.num_tokens {
                context.remove(0);
                context.push(0);
                cnt -= 1;
            }
            context[cnt] = next_ch;
            cnt += 1;
        }
        Ok(chs)
    }
}
