pub trait TensorElement: Clone + Copy + Sized + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn as_f32(self) -> f32;
}

impl TensorElement for f32 {
    fn zero() -> Self {
        0.
    }
    fn one() -> Self {
        1.
    }
    fn as_f32(self) -> f32 {
        self
    }
}

impl TensorElement for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn as_f32(self) -> f32 {
        self as f32
    }
}

impl TensorElement for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
    fn as_f32(self) -> f32 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}
