use crate::funcs::*;
use crate::graph::{Graph, GraphError, TensorId};
use crate::optimizer::{Optimizer, OptimizerState};
use crate::tensor::{GeneralTensor, Tensor, TensorError, TensorOps};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub tensors: HashMap<String, Tensor<f32>>,
    pub optimizer: OptimizerState,
}

pub struct GPT<G: Graph> {
    graph: G,
    num_tokens: usize,
    token_input: TensorId,
    pos_input: TensorId,
    output: TensorId,
    expected_output: TensorId,
    loss: TensorId,
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

impl<G: Graph> GPT<G> {
    pub fn new<R: Rng>(
        rng: &mut R,
        mut g: G,
        batch_size: Option<usize>,
        vocab_size: usize,
        embedding_degree: usize,
        num_tokens: usize,
        num_layers: usize,
        num_heads: usize,
        head_size: usize,
        dropout: f32,
    ) -> Result<Self, GraphError> {
        // Mapping each token to a `embedding_degree` dimension space through a lookup table
        let token_embedding = g.alloc(
            Tensor::<f32>::rand(rng, &[vocab_size, embedding_degree]),
            true,
            "token_embedding".into(),
        )?;

        // Token inputs. We will get `num_tokens` tokens as inputs and will have `num_tokens`
        // outputs.
        // In the case of CPU training, it's much more efficient to parallelize over instances
        // in a single batch. (I.e. it's not very efficient to parallelize a matrix multiplication
        // operation on CPUs. Better approach is to process a 32-instanced batch on a 32-core CPU,
        // where each instance runs on its own core, without parallelizing operations)
        // That's why we DO NOT specify a `batch_size` when training on a CPU.
        let token_input = g.alloc_usize(
            Tensor::<usize>::zeros(&if let Some(batch_size) = batch_size {
                vec![batch_size, num_tokens]
            } else {
                vec![num_tokens]
            }),
            "token_input".into(),
        )?;

        let expected_output = g.alloc_usize(
            Tensor::<usize>::zeros(&if let Some(batch_size) = batch_size {
                vec![batch_size, num_tokens]
            } else {
                vec![num_tokens]
            }),
            "expected_output".into(),
        )?;

        // Map the token index into a `embedding_degree` dimension vector through the `token_embedding`
        // lookup table.
        let embedded_token_input = g.call(Embedding::new(), &[token_input, token_embedding])?;

        // Map token positions into `embedding_degree` dimension vectors.
        let pos_input = g.alloc(
            Tensor::<f32>::rand(rng, &[num_tokens, embedding_degree]),
            false,
            "pos_input".into(),
        )?;

        // Positional+Token information will both reside in a single `embedding_degree` dimension
        // vector.
        let inp = g.call(Add::new(), &[embedded_token_input, pos_input])?;

        let mut curr_inp = inp;
        for l in 0..num_layers {
            // Normalize input before applying multi-head attention
            let norm_coeff = g.alloc(
                Tensor::<f32>::rand(rng, &[embedding_degree]),
                true,
                format!("norm_{}_coeff", l),
            )?;
            let norm_bias = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                true,
                format!("norm_{}_bias", l),
            )?;
            let norm_inp = g.call(LayerNorm::new(), &[curr_inp, norm_coeff, norm_bias])?;

            let mut heads = Vec::new();

            // Multi-head Attention
            for h in 0..num_heads {
                // Key
                let k_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    true,
                    format!("head_{}_{}_k", l, h),
                )?;
                let k = g.call(MatMul::new(), &[norm_inp, k_params])?;

                // Query
                let q_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    true,
                    format!("head_{}_{}_q", l, h),
                )?;
                let q = g.call(MatMul::new(), &[norm_inp, q_params])?;

                // Value
                let v_params = g.alloc(
                    Tensor::<f32>::rand(rng, &[embedding_degree, head_size]),
                    true,
                    format!("head_{}_{}_v", l, h),
                )?;
                let v = g.call(MatMul::new(), &[norm_inp, v_params])?;

                let q_t = g.call(Transpose::new(), &[q])?;
                let kq = g.call(MatMul::new(), &[k, q_t])?;

                let head_size_sqrt_inv = (head_size as f32).powf(-0.5);
                let kq_coeff = g.call(Coeff::new(head_size_sqrt_inv), &[kq])?;

                let masked_kq = g.call(TrilMask::new(num_tokens), &[kq_coeff])?;
                let soft_masked_kq = g.call(Softmax::new(), &[masked_kq])?;
                let dropped_soft_masked_kq = g.call(Dropout::new(dropout), &[soft_masked_kq])?;
                let atten = g.call(MatMul::new(), &[dropped_soft_masked_kq, v])?;
                heads.push(atten);
            }

            // Concat head results and project into embedding_degree
            let cat = g.call(Cat::new(), &heads)?;
            let proj_params = g.alloc(
                Tensor::<f32>::rand(rng, &[num_heads * head_size, embedding_degree]),
                true,
                format!("proj_{}_weights", l),
            )?;
            let proj_bias_params = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                true,
                format!("proj_{}_bias", l),
            )?;
            let proj_cat = g.call(MatMul::new(), &[cat, proj_params])?;
            let proj_cat_bias = g.call(Add::new(), &[proj_cat, proj_bias_params])?;
            let dropped_proj_cat_bias = g.call(Dropout::new(dropout), &[proj_cat_bias])?;

            // Add attention results to input and then normalize
            let add_atten = g.call(Add::new(), &[norm_inp, dropped_proj_cat_bias])?;
            let add_atten_norm_coeff = g.alloc(
                Tensor::<f32>::rand(rng, &[embedding_degree]),
                true,
                format!("atten_norm_{}_coeff", l),
            )?;
            let add_atten_norm_bias = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                true,
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
                true,
                format!("feedforward1_{}_weights", l),
            )?;
            let bias1_params = g.alloc(
                Tensor::<f32>::zeros(&[4 * embedding_degree]),
                true,
                format!("feedforward1_{}_bias", l),
            )?;
            let lin1_result = g.call(MatMul::new(), &[add_atten_norm, lin1_params])?;
            let lin1_bias_result = g.call(Add::new(), &[lin1_result, bias1_params])?;
            let lin1_act = g.call(Relu::new(), &[lin1_bias_result])?;
            let lin2_params = g.alloc(
                Tensor::<f32>::rand(rng, &[4 * embedding_degree, embedding_degree]),
                true,
                format!("feedforward2_{}_weights", l),
            )?;
            let bias2_params = g.alloc(
                Tensor::<f32>::zeros(&[embedding_degree]),
                true,
                format!("feedforward2_{}_bias", l),
            )?;
            let lin2_result = g.call(MatMul::new(), &[lin1_act, lin2_params])?;
            let lin2_bias_result = g.call(Add::new(), &[lin2_result, bias2_params])?;

            curr_inp = g.call(Add::new(), &[add_atten_norm, lin2_bias_result])?;
        }

        // Normalize the output after the last layer
        let norm_out_coeff = g.alloc(
            Tensor::<f32>::rand(rng, &[embedding_degree]),
            true,
            format!("head_norm_coeff"),
        )?;
        let norm_out_bias = g.alloc(
            Tensor::<f32>::zeros(&[embedding_degree]),
            true,
            format!("head_norm_bias"),
        )?;
        let norm_out = g.call(LayerNorm::new(), &[curr_inp, norm_out_coeff, norm_out_bias])?;

        // Map from embedding_degree to vocab_size through a linear layer
        let to_vocab = g.alloc(
            Tensor::<f32>::rand(rng, &[embedding_degree, vocab_size]),
            true,
            format!("head_map_weights"),
        )?;
        let to_vocab_bias = g.alloc(
            Tensor::<f32>::zeros(&[vocab_size]),
            true,
            format!("head_map_bias"),
        )?;
        let result_lin = g.call(MatMul::new(), &[norm_out, to_vocab])?;
        let output = g.call(Add::new(), &[result_lin, to_vocab_bias])?;

        let loss = g.call(CrossEntropyFunc::new(), &[output, expected_output])?;

        Ok(Self {
            graph: g,
            num_tokens,
            token_input,
            pos_input,
            output,
            expected_output,
            loss,
            pos_input_fixed: pos_encode_inter(num_tokens, embedding_degree),
        })
    }

    pub fn sync(&mut self) -> Result<(), GraphError> {
        self.graph
            .params()
            .to_vec()
            .into_iter()
            .map(|p| self.graph.fetch(p, false))
            .collect::<Result<Vec<_>, GraphError>>()?;
        Ok(())
    }

    pub fn num_params(&self) -> usize {
        self.graph
            .params()
            .to_vec()
            .into_iter()
            .map(|p| self.graph.get(p).unwrap().as_float().unwrap().size())
            .sum::<usize>()
    }

    pub fn set_training_state(
        &mut self,
        training_state: TrainingState,
        load_optimizer: bool,
    ) -> Result<(), GraphError> {
        for p in self.graph.params().to_vec() {
            let name = self.graph.name_of(p)?;
            if let Some(t) = training_state.tensors.get(name) {
                self.graph.load(p, t)?;
            }
        }
        if load_optimizer {
            self.graph.set_optimizer_state(&training_state.optimizer)?;
        }
        Ok(())
    }

    pub fn get_training_state(&self) -> Result<TrainingState, GraphError> {
        let mut state = TrainingState {
            tensors: Default::default(),
            optimizer: self.graph.get_optimizer_state()?,
        };
        for p in self.graph.params().iter() {
            let k = self.graph.name_of(*p)?.to_string();
            let v = self.graph.get(*p)?.as_float()?.clone();
            state.tensors.insert(k, v);
        }
        Ok(state)
    }

    pub fn train_cpu<
        O: Optimizer,
        F: Fn(usize) -> f32,
        C: Fn(&mut Self) -> Result<(), GraphError>,
    >(
        &mut self,
        dataset: &[usize],
        num_batches: usize,
        batch_size: usize,
        limit: Option<usize>,
        optimizer: &O,
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

                    graph.load_usize(self.token_input, &xs)?;
                    graph.load_usize(self.expected_output, &ys)?;
                    graph.forward(true)?;
                    graph.zero_grad()?;
                    let err = graph.backward_all(self.loss, limit)?;
                    Ok((graph, err))
                })
                .collect::<Result<Vec<(G, f32)>, GraphError>>()?
                .into_iter()
                .unzip();
            for (id, avg) in self
                .graph
                .params()
                .to_vec()
                .into_iter()
                .map(|id| {
                    let mut avg = Tensor::<f32>::scalar(0.);
                    for g in graphs.iter() {
                        avg = (&avg + g.get_grad(id)?)?;
                    }
                    avg = avg.map_values(|f| f / graphs.len() as f32);
                    Ok((id, avg))
                })
                .collect::<Result<Vec<_>, GraphError>>()?
            {
                self.graph.load_grad(id, &avg)?;
            }
            let avg_loss = errs.iter().sum::<f32>() / errs.len() as f32;
            let lr = learning_rate(self.graph.optimizer_step());
            self.graph.optimize(optimizer, lr)?;
            if i % 10 == 0 {
                self.sync()?;
                callback(self)?;
            }
            println!(
                "Step: {} Loss: {} (Elapsed: {}ms)",
                self.graph.optimizer_step(),
                avg_loss,
                timer.elapsed().as_millis()
            );
        }
        Ok(())
    }

    pub fn train<O: Optimizer, F: Fn(usize) -> f32, C: Fn(&mut Self) -> Result<(), GraphError>>(
        &mut self,
        dataset: &[usize],
        num_batches: usize,
        batch_size: usize,
        limit: Option<usize>,
        optimizer: &O,
        learning_rate: F,
        callback: C,
    ) -> Result<(), GraphError> {
        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        for i in 0..num_batches {
            let timer = Instant::now();
            let mut rng = rand::thread_rng();
            let (xs, ys) = sample_dataset(dataset, batch_size, self.num_tokens, &mut rng);

            self.graph.load_usize(self.token_input, &xs)?;
            self.graph.load_usize(self.expected_output, &ys)?;

            self.graph.forward(true)?;
            self.graph.zero_grad()?;
            let err = self.graph.backward_all(self.loss, limit)?;
            let lr = learning_rate(self.graph.optimizer_step());
            self.graph.optimize(optimizer, lr)?;
            if i % 50 == 0 {
                callback(self)?;
            }
            println!(
                "Step: {} Loss: {} (Elapsed: {}ms)",
                self.graph.optimizer_step(),
                err,
                timer.elapsed().as_millis()
            );
        }
        Ok(())
    }

    pub fn infer<R: Rng, F: Fn(usize) -> ()>(
        &mut self,
        rng: &mut R,
        prompt: &[usize],
        count: usize,
        temperature: f32,
        callback: F,
    ) -> Result<Vec<usize>, GraphError> {
        let mut cnt = prompt.len();
        let mut context = vec![0; self.num_tokens];
        context[..prompt.len()].copy_from_slice(prompt);

        self.graph.load(self.pos_input, &self.pos_input_fixed)?;

        for ch in prompt {
            callback(*ch);
        }
        let mut chs = prompt.to_vec();
        for _ in 0..count {
            self.graph.load_usize(
                self.token_input,
                &Tensor::raw(&[1, self.num_tokens], context.clone())?,
            )?;

            self.graph.forward(false)?;
            self.graph.fetch(self.output, false)?;
            let next_ch = select(
                rng,
                &self
                    .graph
                    .get(self.output)?
                    .as_float()?
                    .get(0)?
                    .get(cnt - 1)?,
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
