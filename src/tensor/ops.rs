use super::*;
use rayon::prelude::*;

impl<V: TensorElement + std::ops::Add<Output = V>> Add for &Tensor<V> {
    type Output = Tensor<V>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        &self.view() + &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn add(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() + other
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        self + &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn add(self, other: &TensorView<V>) -> Self::Output {
        combine_map(self, other, 0, |a, b| {
            Tensor::scalar(a.scalar() + b.scalar())
        })
    }
}

impl<V: TensorElement + std::ops::Sub<Output = V>> Sub for &Tensor<V> {
    type Output = Tensor<V>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        &self.view() - &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn sub(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() - other
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        self - &other.view()
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

impl<V: TensorElement + std::ops::Mul<Output = V>> Mul for &Tensor<V> {
    type Output = Tensor<V>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        &self.view() * &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn mul(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() * other
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        self * &other.view()
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
impl<V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>> BitXor
    for &Tensor<V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        &self.view() ^ &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>>
    BitXor<&TensorView<'a, V>> for &Tensor<V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() ^ other
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>>
    BitXor<&Tensor<V>> for &TensorView<'a, V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        self ^ &other.view()
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
                        sum = sum + a.get(i).get(k).scalar() * b.get(k).get(j).scalar();
                    }
                    sum
                })
                .collect();
            Tensor::raw(&[a.shape()[0], b.shape()[1]], data)
        })
    }
}

impl<V: TensorElement + std::ops::Neg<Output = V>> Neg for &Tensor<V> {
    type Output = Tensor<V>;
    fn neg(self) -> Self::Output {
        -&self.view()
    }
}
impl<'a, V: TensorElement + std::ops::Neg<Output = V>> Neg for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn neg(self) -> Self::Output {
        self.map_values(|f| -f)
    }
}

impl Not for &Tensor<bool> {
    type Output = Tensor<bool>;

    fn not(self) -> Self::Output {
        !&self.view()
    }
}
impl<'a> Not for &TensorView<'a, bool> {
    type Output = Tensor<bool>;

    fn not(self) -> Self::Output {
        Tensor {
            blob: self.blob().iter().map(|b| !b).collect(),
            shape: self.shape().to_vec(),
        }
    }
}
