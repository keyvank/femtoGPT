mod elements;
mod error;
mod helper;
mod ops;
mod view;
pub use elements::*;
pub use error::*;
pub use helper::*;
pub use ops::*;
pub use view::*;

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::ops::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<V: TensorElement> {
    blob: Vec<V>,
    shape: Vec<usize>,
}

impl<T: TensorOps<bool>> From<&T> for Tensor<f32> {
    fn from(v: &T) -> Self {
        v.map_values(|v| v.as_f32())
    }
}

unsafe impl<V: TensorElement> Send for Tensor<V> {}
unsafe impl<V: TensorElement> Sync for Tensor<V> {}

pub fn reshape(size: usize, shape: &[usize]) -> Vec<usize> {
    let mut final_shape = shape.to_vec();
    if shape[0] == 0 && shape[1..].iter().all(|s| *s != 0) {
        let mul = shape[1..].iter().fold(1, |c, s| c * s);
        final_shape[0] = size / mul;
    } else if shape[shape.len() - 1] == 0 && shape[0..shape.len() - 1].iter().all(|s| *s != 0) {
        let mul = shape[..shape.len() - 1].iter().fold(1, |c, s| c * s);
        final_shape[shape.len() - 1] = size / mul;
    } else {
        assert!(shape.iter().all(|s| *s != 0));
    };
    final_shape
}

pub trait TensorMutOps<V: TensorElement>: TensorOps<V> {
    fn blob_mut(&mut self) -> &mut [V];
    fn tensor_mut(&mut self) -> &mut Tensor<V>;

    fn fill(&mut self, v: V) {
        self.blob_mut().fill(v);
    }
    fn set<T: TensorOps<V>>(&mut self, t: T) -> Result<(), TensorError> {
        if self.shape() != t.shape() {
            return Err(TensorError::ShapeError);
        }
        self.blob_mut().clone_from_slice(t.blob());
        Ok(())
    }
    fn get_mut(&mut self, ind: usize) -> TensorMutView<V> {
        let sub_size = self.size() / self.len();
        TensorMutView {
            offset: self.offset() + sub_size * ind,
            shape: self.shape()[1..].to_vec(),
            mirror: self.tensor_mut(),
        }
    }
}

pub trait TensorOps<V: TensorElement>: Sized + Into<Tensor<V>> + Send + Sync {
    fn shape(&self) -> &[usize];
    fn blob(&self) -> &[V];
    fn tensor(&self) -> &Tensor<V>;
    fn offset(&self) -> usize;

    fn mean(&self) -> f32
    where
        V: std::iter::Sum,
    {
        self.blob().iter().cloned().sum::<V>().as_f32() / self.size() as f32
    }

    fn keep_right(&self, dims: usize) -> Result<TensorView<V>, TensorError> {
        let mut new_shape = self.shape().to_vec();
        if self.dim() == dims {
            new_shape.insert(0, 1);
        }
        for _i in 0..new_shape.len() - dims - 1 {
            let rem = new_shape.remove(0);
            new_shape[0] *= rem;
        }
        self.reshape(&new_shape)
    }

    fn map_values<W: TensorElement, F: Fn(V) -> W + Sync + Send>(&self, f: F) -> Tensor<W> {
        Tensor {
            blob: self.blob().iter().map(|v| f(*v)).collect::<Vec<_>>(),
            shape: self.shape().to_vec(),
        }
    }

    fn map<
        W: TensorElement,
        F: Fn(TensorView<V>) -> Result<Tensor<W>, TensorError> + Sync + Send,
    >(
        &self,
        dim: usize,
        f: F,
    ) -> Result<Tensor<W>, TensorError> {
        let blob = self
            .keep_right(dim)?
            .inners()
            .into_iter()
            .map(|v| f(v))
            .collect::<Result<Vec<_>, TensorError>>()?;
        if !blob.iter().all(|t| t.shape() == blob[0].shape()) {
            return Err(TensorError::InconsistentShapes);
        }
        let mut out_shape = self.shape()[..self.dim() - dim].to_vec();
        out_shape.extend(blob[0].shape());
        Ok(Tensor {
            blob: blob.into_iter().map(|t| t.blob).flatten().collect(),
            shape: out_shape,
        })
    }

    fn scalar(&self) -> Result<V, TensorError> {
        if self.dim() == 0 {
            Ok(self.blob()[0])
        } else {
            Err(TensorError::NotScalar)
        }
    }
    fn inners<'a>(&'a self) -> Vec<TensorView<'a, V>> {
        (0..self.len()).map(|i| self.get(i)).collect::<Vec<_>>()
    }
    fn dim(&self) -> usize {
        self.shape().len()
    }
    fn len(&self) -> usize {
        *self
            .shape()
            .get(0)
            .expect("Scalar values don't have a size!")
    }
    fn size(&self) -> usize {
        self.shape().iter().fold(1, |curr, s| curr * s)
    }
    fn view(&self) -> TensorView<V> {
        TensorView {
            mirror: self.tensor(),
            offset: self.offset(),
            shape: self.shape().to_vec(),
        }
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorView<V>, TensorError> {
        let final_shape = reshape(self.size(), shape);
        let new_size = final_shape.iter().fold(1, |c, s| c * s);
        if new_size != self.size() {
            return Err(TensorError::ShapeError);
        }
        let offset = self.offset();
        Ok(TensorView {
            mirror: self.tensor(),
            offset: offset,
            shape: final_shape,
        })
    }

    fn get(&self, ind: usize) -> TensorView<V> {
        let sub_size = self.size() / self.len();
        TensorView {
            offset: self.offset() + sub_size * ind,
            shape: self.shape()[1..].to_vec(),
            mirror: self.tensor(),
        }
    }

    fn transpose(&self) -> Result<Tensor<V>, TensorError> {
        self.map(2, |m| {
            let d0 = m.shape()[0];
            let d1 = m.shape()[1];
            let mut dat = Vec::with_capacity(d1 * d0);
            let m_blob = m.blob();
            for j in 0..d1 {
                for i in 0..d0 {
                    dat.push(m_blob[i * d1 + j]);
                }
            }
            Ok(Tensor::raw(&[d1, d0], dat))
        })
    }
}

impl<V: TensorElement> TensorMutOps<V> for Tensor<V> {
    fn tensor_mut(&mut self) -> &mut Tensor<V> {
        self
    }
    fn blob_mut(&mut self) -> &mut [V] {
        &mut self.blob
    }
}

impl<V: TensorElement> TensorOps<V> for Tensor<V> {
    fn tensor(&self) -> &Tensor<V> {
        self
    }

    fn offset(&self) -> usize {
        0
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        &self.blob
    }
}

impl<V: TensorElement> Tensor<V> {
    pub fn raw(shape: &[usize], blob: Vec<V>) -> Self {
        let sz = shape.iter().fold(1, |c, s| c * s);
        assert_eq!(sz, blob.len());
        Self {
            blob,
            shape: shape.to_vec(),
        }
    }
    pub fn tril(n: usize) -> Self {
        Tensor {
            blob: (0..n * n)
                .map(|i| if i % n <= i / n { V::one() } else { V::zero() })
                .collect(),
            shape: vec![n, n],
        }
    }
    pub fn scalar(v: V) -> Self {
        Tensor {
            blob: vec![v],
            shape: vec![],
        }
    }
    pub fn vector(v: &[V]) -> Self {
        Tensor {
            blob: v.to_vec(),
            shape: vec![v.len()],
        }
    }
    pub fn constant(shape: &[usize], value: V) -> Self {
        Tensor {
            blob: vec![value; shape.iter().fold(1, |curr, s| curr * s)],
            shape: shape.to_vec(),
        }
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self::constant(shape, V::zero())
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self::constant(shape, V::one())
    }
    pub fn rand_range<R: Rng>(r: &mut R, start: f32, end: f32, shape: &[usize]) -> Tensor<f32> {
        Tensor::<f32> {
            blob: (0..shape.iter().fold(1, |curr, s| curr * s))
                .map(|_| r.gen_range(start..end))
                .collect(),
            shape: shape.to_vec(),
        }
    }
    pub fn rand<R: Rng>(r: &mut R, shape: &[usize]) -> Tensor<f32> {
        let normal = Normal::new(0.0, 0.02).unwrap();
        Tensor::<f32> {
            blob: (0..shape.iter().fold(1, |curr, s| curr * s))
                .map(|_| normal.sample(r))
                .collect(),
            shape: shape.to_vec(),
        }
    }
    pub fn cat<T: TensorOps<V>>(inps: &[&T]) -> Self {
        let shape = inps
            .get(0)
            .expect("No tensors to be concatenated!")
            .shape()
            .to_vec();
        inps.iter().all(|t| t.shape() == shape);
        let each_sz = inps.get(0).unwrap().size();
        let group_size = shape.last().unwrap();
        let mut offset = 0;
        let mut data: Vec<V> = Vec::with_capacity(each_sz * inps.len());

        while offset < each_sz {
            for inp in inps {
                data.extend(&inp.blob()[offset..offset + group_size]);
            }
            offset += group_size;
        }

        let mut target_shape = shape.clone();
        target_shape[shape.len() - 1] = target_shape[shape.len() - 1] * inps.len();
        Tensor::raw(&target_shape, data)
    }
    pub fn split<T: TensorOps<V>>(inp: &T, cnt: usize) -> Vec<Tensor<V>> {
        let group_size = inp.shape().last().unwrap() / cnt;
        let mut result = vec![Vec::<V>::new(); cnt];
        let mut offset = 0;
        let inp_blob = inp.blob();
        while offset < inp.size() {
            for i in 0..cnt {
                result[i].extend(&inp_blob[offset..offset + group_size]);
                offset += group_size;
            }
        }
        let mut target_shape = inp.shape().to_vec();
        target_shape[inp.dim() - 1] = target_shape[inp.dim() - 1] / cnt;
        result
            .into_iter()
            .map(|d| Tensor::raw(&target_shape, d))
            .collect()
    }
}
