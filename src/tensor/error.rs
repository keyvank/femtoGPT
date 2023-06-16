use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("unexpected tensor type!")]
    UnexpectedType,
    #[error("unexpected tensor shape!")]
    UnexpectedShape,
    #[error("invalid index!")]
    InvalidIndex,
}
