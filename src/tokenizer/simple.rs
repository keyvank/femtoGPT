use super::Tokenizer;
use std::collections::{HashMap, HashSet};

/// A minimal **character-level tokenizer** for converting between raw text and integer tokens.
///
/// The tokenizer scans a dataset to identify all **unique characters**, then builds two
/// bidirectional lookup maps:
///
/// - `ch_to_int`: maps each character → integer token ID  
/// - `int_to_ch`: maps each integer token ID → character
///
/// This ensures a deterministic and reversible mapping between text and token sequences.
/// It is intentionally simple and efficient for small-scale or experimental GPT models.
///
/// # Example
/// ```ignore
/// let dataset = "Hello";
/// let tokenizer = SimpleTokenizer::new(dataset);
///
/// Vocabulary size = 4 unique characters: {'H', 'e', 'l', 'o'}
/// assert_eq!(tokenizer.vocab_size(), 4);
/// ```
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
    /// Returns the total number of unique characters in the tokenizer vocabulary.
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    /// Encodes a string into integer tokens using the internal `ch_to_int` mapping.
    ///
    /// # Panics
    /// This function will panic if the string contains a character **not present**
    /// in the tokenizer vocabulary.
    ///
    /// # Example
    /// ```example
    /// let tokenizer = SimpleTokenizer::new("abc");
    /// assert_eq!(tokenizer.tokenize("cab"), vec![2, 0, 1]);
    /// ```
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .chars()
            .map(|ch| self.ch_to_int.get(&ch).unwrap().clone())
            .collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|tkn| self.int_to_ch.get(tkn).unwrap().clone())
            .collect()
    }
}
