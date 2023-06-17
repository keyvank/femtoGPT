pub trait TensorElement: Clone + Copy + Sized + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
}

impl TensorElement for f32 {
    fn zero() -> Self {
        0.
    }
    fn one() -> Self {
        1.
    }
}

impl TensorElement for usize {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
