use super::*;

pub fn reshape(size: usize, shape: &[usize]) -> Vec<usize> {
    let mut final_shape = shape.to_vec();
    if shape[0] == 0 && shape[1..].iter().all(|s| *s != 0) {
        let mul = shape[1..].iter().fold(1, |c, s| c * s);
        final_shape[0] = size / mul;
    } else if shape[shape.len() - 1] == 0 && shape[0..shape.len() - 1].iter().all(|s| *s != 0) {
        let mul = shape[..shape.len() - 1].iter().fold(1, |c, s| c * s);
        final_shape[shape.len() - 1] = size / mul;
    } else {
        assert!(shape.iter().all(|s| *s != 0));
    };
    final_shape
}
