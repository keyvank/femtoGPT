use super::Tokenizer;
use std::collections::{HashMap, HashSet};

pub struct SimpleTokenizer {
    vocab_size: usize,
    ch_to_int: HashMap<char, usize>,
    int_to_ch: HashMap<usize, char>,
}

impl SimpleTokenizer {
    pub fn new(dataset: &str) -> Self {
        let mut chars = dataset
            .chars()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        chars.sort();
        let int_to_ch = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (i, *ch))
            .collect::<HashMap<usize, char>>();
        let ch_to_int = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i))
            .collect::<HashMap<char, usize>>();
        Self {
            vocab_size: chars.len(),
            int_to_ch,
            ch_to_int,
        }
    }
}

impl Tokenizer for SimpleTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    //tokennize的作用是将字符串转换为token
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .chars()
            .map(|ch| self.ch_to_int.get(&ch).unwrap().clone())
            .collect()
    }
    //untokenize的作用是将token转换为字符串
    fn untokenize(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|tkn| self.int_to_ch.get(tkn).unwrap().clone())
            .collect()
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn test_simple_tokenizer(){
        let dataset = "abc";
        let tokenizer = SimpleTokenizer::new(dataset);
        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.tokenize("a"), vec![0]);
        assert_eq!(tokenizer.tokenize("b"), vec![1]);
        assert_eq!(tokenizer.tokenize("c"), vec![2]);
        assert_eq!(tokenizer.tokenize("abc"), vec![0, 1, 2]);
        assert_eq!(tokenizer.untokenize(&vec![0, 1, 2]), "abc");
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = SimpleTokenizer::new("hello world");
        let text = "hello"; 
        let tokens = tokenizer.tokenize(text);
        let decoded = tokenizer.untokenize(&tokens); 

        assert_eq!(decoded, text);  
    }
}
