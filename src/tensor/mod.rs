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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GeneralTensor {
    Float(Tensor<f32>),
    Usize(Tensor<usize>),
}

impl GeneralTensor {
    pub fn size(&self) -> usize {
        match self {
            GeneralTensor::Float(t) => t.size(),
            GeneralTensor::Usize(t) => t.size(),
        }
    }
    pub fn shape(&self) -> &[usize] {
        match self {
            GeneralTensor::Float(t) => t.shape(),
            GeneralTensor::Usize(t) => t.shape(),
        }
    }
    pub fn as_float(&self) -> Result<&Tensor<f32>, TensorError> {
        match self {
            GeneralTensor::Float(t) => Ok(t),
            _ => Err(TensorError::UnexpectedType),
        }
    }
    pub fn as_usize(&self) -> Result<&Tensor<usize>, TensorError> {
        match self {
            GeneralTensor::Usize(t) => Ok(t),
            _ => Err(TensorError::UnexpectedType),
        }
    }
    pub fn as_float_mut(&mut self) -> Result<&mut Tensor<f32>, TensorError> {
        match self {
            GeneralTensor::Float(t) => Ok(t),
            _ => Err(TensorError::UnexpectedType),
        }
    }
    pub fn as_usize_mut(&mut self) -> Result<&mut Tensor<usize>, TensorError> {
        match self {
            GeneralTensor::Usize(t) => Ok(t),
            _ => Err(TensorError::UnexpectedType),
        }
    }
}

impl<V: TensorElement> Tensor<V> {
    pub fn raw(shape: &[usize], blob: Vec<V>) -> Result<Self, TensorError> {
        let sz = shape.iter().fold(1, |c, s| c * s);
        if sz != blob.len() {
            return Err(TensorError::UnexpectedShape);
        }
        Ok(Self {
            blob,
            shape: shape.to_vec(),
        })
    }
    pub fn scalar(v: V) -> Self {
        Tensor {
            blob: vec![v],
            shape: vec![],
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
}

unsafe impl<V: TensorElement> Send for Tensor<V> {}
unsafe impl<V: TensorElement> Sync for Tensor<V> {}

pub trait TensorMutOps<V: TensorElement>: TensorOps<V> {
    fn blob_mut(&mut self) -> &mut [V];
    fn tensor_mut(&mut self) -> &mut Tensor<V>;

    fn fill(&mut self, v: V) {
        self.blob_mut().fill(v);
    }
    fn set<T: TensorOps<V>>(&mut self, t: T) -> Result<(), TensorError> {
        if self.shape() != t.shape() {
            return Err(TensorError::UnexpectedShape);
        }
        self.blob_mut().clone_from_slice(t.blob());
        Ok(())
    }
    fn get_mut(&mut self, ind: usize) -> Result<TensorMutView<V>, TensorError> {
        if ind >= self.len() {
            return Err(TensorError::InvalidIndex);
        }
        let sub_size = self.size() / self.len();
        Ok(TensorMutView {
            offset: self.offset() + sub_size * ind,
            shape: self.shape()[1..].to_vec(),
            mirror: self.tensor_mut(),
        })
    }
}

impl Tensor<f32> {
    pub fn mean(&self) -> f32 {
        self.blob().iter().cloned().sum::<f32>() / self.size() as f32
    }
}

pub trait TensorOps<V: TensorElement>: Sized + Into<Tensor<V>> + Send + Sync {
    fn shape(&self) -> &[usize];
    fn blob(&self) -> &[V];
    fn tensor(&self) -> &Tensor<V>;
    fn offset(&self) -> usize;

    fn keep_right(&self, dims: usize) -> Result<TensorView<V>, TensorError> {
        let mut shape = self.shape().to_vec();
        if shape.len() < dims {
            return Err(TensorError::UnexpectedShape);
        } else if shape.len() == dims {
            shape.insert(0, 1);
        } else {
            while shape.len() > dims + 1 {
                let rem = shape.remove(0);
                shape[0] *= rem;
            }
        }
        Ok(TensorView {
            mirror: self.tensor(),
            offset: self.offset(),
            shape,
        })
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
            return Err(TensorError::UnexpectedShape);
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
            Err(TensorError::UnexpectedShape)
        }
    }
    fn inners<'a>(&'a self) -> Vec<TensorView<'a, V>> {
        (0..self.len())
            .map(|i| self.get(i).unwrap())
            .collect::<Vec<_>>()
    }
    fn dim(&self) -> usize {
        self.shape().len()
    }
    fn len(&self) -> usize {
        *self.shape().get(0).unwrap_or(&0) // Scalar has a len of 0
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

    fn get(&self, ind: usize) -> Result<TensorView<V>, TensorError> {
        if ind >= self.len() {
            return Err(TensorError::InvalidIndex);
        }
        let sub_size = self.size() / self.len();
        Ok(TensorView {
            offset: self.offset() + sub_size * ind,
            shape: self.shape()[1..].to_vec(),
            mirror: self.tensor(),
        })
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
            Ok(Tensor {
                blob: dat,
                shape: [d1, d0].to_vec(),
            })
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
