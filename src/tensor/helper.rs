use super::*;

impl<V: TensorElement + std::ops::Add<Output = V>> Add for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        &self.view() + &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn add(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() + other
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        self + &other.view()
    }
}
impl<V: TensorElement + std::ops::Sub<Output = V>> Sub for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        &self.view() - &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn sub(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() - other
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        self - &other.view()
    }
}
impl<V: TensorElement + std::ops::Mul<Output = V>> Mul for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        &self.view() * &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn mul(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() * other
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        self * &other.view()
    }
}
impl<
        V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V> + std::ops::AddAssign,
    > BitXor for &Tensor<V>
{
    type Output = Result<Tensor<V>, TensorError>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        &self.view() ^ &other.view()
    }
}
impl<
        'a,
        V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V> + std::ops::AddAssign,
    > BitXor<&TensorView<'a, V>> for &Tensor<V>
{
    type Output = Result<Tensor<V>, TensorError>;
    fn bitxor(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() ^ other
    }
}
impl<
        'a,
        V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V> + std::ops::AddAssign,
    > BitXor<&Tensor<V>> for &TensorView<'a, V>
{
    type Output = Result<Tensor<V>, TensorError>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        self ^ &other.view()
    }
}
impl Not for &Tensor<bool> {
    type Output = Tensor<bool>;

    fn not(self) -> Self::Output {
        !&self.view()
    }
}
