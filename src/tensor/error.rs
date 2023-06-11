use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("unexpected tensor shape!")]
    UnexpectedShape,
}
