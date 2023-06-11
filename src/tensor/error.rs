use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("incompatible tensor shapes!")]
    ShapeError,
    #[error("map function is returning inconsistent shapes!")]
    InconsistentShapes,
    #[error("not a scalar!")]
    NotScalar,
}
