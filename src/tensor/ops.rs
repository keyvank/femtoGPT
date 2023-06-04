use super::*;

impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn add(self, other: &TensorView<V>) -> Self::Output {
        combine_map(self, other, 0, |a, b| {
            Tensor::scalar(a.scalar() + b.scalar())
        })
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn sub(self, other: &TensorView<V>) -> Self::Output {
        combine_map(self, other, 0, |a, b| {
            Tensor::scalar(a.scalar() - b.scalar())
        })
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn mul(self, other: &TensorView<V>) -> Self::Output {
        combine_map(self, other, 0, |a, b| {
            Tensor::scalar(a.scalar() * b.scalar())
        })
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>> BitXor
    for &TensorView<'a, V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &TensorView<V>) -> Self::Output {
        combine_map(self, other, 2, |a, b| {
            let works = a.shape()[0] * b.shape()[1];
            let data = (0..works)
                .into_par_iter()
                .map(|work| {
                    let j = work % b.shape()[1];
                    let i = work / b.shape()[1];
                    let mut sum = <<V as std::ops::Mul>::Output as std::ops::Add>::Output::zero();
                    for k in 0..a.shape()[1] {
                        sum = sum + a.blob()[i * a.shape()[1] + k] * b.blob()[k * b.shape()[1] + j];
                    }
                    sum
                })
                .collect();
            Tensor::raw(&[a.shape()[0], b.shape()[1]], data)
        })
    }
}
impl<'a> Not for &TensorView<'a, bool> {
    type Output = Tensor<bool>;

    fn not(self) -> Self::Output {
        self.map_values(|b| !b)
    }
}
