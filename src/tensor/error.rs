use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("incompatible tensor shapes!")]
    ShapeError,
}
