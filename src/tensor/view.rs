use super::*;

#[derive(Debug, Clone)]
pub struct TensorView<'a, V: TensorElement> {
    pub(super) mirror: &'a Tensor<V>,
    pub(super) offset: usize,
    pub(super) shape: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorMutView<'a, V: TensorElement> {
    pub(super) mirror: &'a mut Tensor<V>,
    pub(super) offset: usize,
    pub(super) shape: Vec<usize>,
}

impl<V: TensorElement> From<TensorView<'_, V>> for Tensor<V> {
    fn from(view: TensorView<'_, V>) -> Tensor<V> {
        Tensor {
            blob: view.blob().to_vec(),
            shape: view.shape().to_vec(),
        }
    }
}

impl<V: TensorElement> From<TensorMutView<'_, V>> for Tensor<V> {
    fn from(view: TensorMutView<'_, V>) -> Tensor<V> {
        view.into()
    }
}

impl<V: TensorElement> TensorMutOps<V> for TensorMutView<'_, V> {
    fn tensor_mut(&mut self) -> &mut Tensor<V> {
        &mut self.mirror
    }
    fn blob_mut(&mut self) -> &mut [V] {
        let sz = self.size();
        &mut self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl<V: TensorElement> TensorOps<V> for TensorMutView<'_, V> {
    fn tensor(&self) -> &Tensor<V> {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        let sz = self.size();
        &self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl<V: TensorElement> TensorOps<V> for TensorView<'_, V> {
    fn tensor(&self) -> &Tensor<V> {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        &self.mirror.blob[self.offset..self.offset + self.size()]
    }
}
