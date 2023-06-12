use super::*;

pub fn binary<
    'a,
    V: TensorElement,
    W: TensorElement,
    T1: TensorOps<V>,
    T2: TensorOps<V>,
    F: Fn(V, V) -> W + Sync + Send,
>(
    a: &T1,
    b: &T2,
    f: F,
) -> Result<Tensor<W>, TensorError> {
    let (a, b, rev) = if a.dim() > b.dim() {
        (a.view(), b.view(), false)
    } else {
        (b.view(), a.view(), true)
    };
    a.map(b.dim(), |a| {
        assert_eq!(a.shape(), b.shape());
        let (a, b) = if rev { (&b, &a) } else { (&a, &b) };
        Tensor::raw(
            a.shape(),
            a.blob()
                .iter()
                .zip(b.blob().iter())
                .map(|(a, b)| f(*a, *b))
                .collect(),
        )
    })
}

impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn add(self, other: &TensorView<V>) -> Self::Output {
        binary(self, other, |a, b| a + b)
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn sub(self, other: &TensorView<V>) -> Self::Output {
        binary(self, other, |a, b| a - b)
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul for &TensorView<'a, V> {
    type Output = Result<Tensor<V>, TensorError>;
    fn mul(self, other: &TensorView<V>) -> Self::Output {
        binary(self, other, |a, b| a * b)
    }
}
impl<
        'a,
        V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V> + std::ops::AddAssign,
    > BitXor for &TensorView<'a, V>
{
    type Output = Result<Tensor<V>, TensorError>;
    fn bitxor(self, other: &TensorView<V>) -> Self::Output {
        let (a, b, rev) = if self.dim() > other.dim() {
            (self.view(), other.view(), false)
        } else {
            (other.view(), self.view(), true)
        };
        a.map(b.dim(), |a| {
            assert_eq!(a.shape()[..a.dim() - 2], b.shape()[..b.dim() - 2]);
            let (a, b) = if rev { (&b, &a) } else { (&a, &b) };
            let data = a
                .keep_right(2)?
                .inners()
                .iter()
                .zip(b.keep_right(2)?.inners().iter())
                .map(|(a, b)| {
                    let m = a.shape()[0];
                    let n = a.shape()[1];
                    let p = b.shape()[1];
                    let mut result = vec![
                            <<V as std::ops::Mul>::Output as std::ops::Add>::Output::zero();
                            m * p
                        ];
                    let a_blob = a.blob();
                    let b_blob = b.blob();
                    for i in 0..m {
                        for k in 0..n {
                            for j in 0..p {
                                result[i * p + j] += a_blob[i * n + k] * b_blob[k * p + j];
                            }
                        }
                    }
                    result
                })
                .flatten()
                .collect();
            let mut final_shape = a.shape().to_vec();
            final_shape[a.dim() - 1] = b.shape()[b.dim() - 1];
            Tensor::raw(&final_shape, data)
        })
    }
}
impl<'a> Not for &TensorView<'a, bool> {
    type Output = Tensor<bool>;

    fn not(self) -> Self::Output {
        self.map_values(|b| !b)
    }
}
