// Copyright 2019 Google LLC. All Rights Reserved.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2023 Keyvan Kambakhsh
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::Tokenizer;

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const PREFIXED_UNDERSCORE: char = '\u{2581}';

#[derive(Clone, Copy)]
struct Node {
    index: usize,
    start: usize,
}

struct DagNode {
    text: String,
    len: usize,
    score: f32,
    index: usize,
    leaf: bool,
    children: HashMap<char, DagNode>,
}

impl DagNode {
    fn new(text: String) -> DagNode {
        let len = text.chars().count();
        DagNode {
            text,
            len,
            score: 0.0,
            index: 0,
            leaf: false,
            children: HashMap::new(),
        }
    }
}

pub struct SentencePieceTokenizer {
    root: DagNode,
    vocab: Vec<String>,
}

impl SentencePieceTokenizer {
    pub fn load<P: AsRef<Path>>(vocab_file: P) -> io::Result<SentencePieceTokenizer> {
        let mut model = SentencePieceTokenizer {
            root: DagNode::new("".to_string()),
            vocab: Default::default(),
        };

        let f = File::open(vocab_file)?;
        let reader = BufReader::new(f);
        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let split = line.splitn(2, "\t").collect::<Vec<_>>();
            if split.len() != 2 {
                return Err(io::Error::from(io::ErrorKind::InvalidData));
            }

            let token = split[0];
            let score = split[1]
                .parse::<f32>()
                .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;

            model.insert(token, score, idx);
        }
        Ok(model)
    }

    fn insert(&mut self, word: &str, score: f32, index: usize) {
        self.vocab.insert(index as usize, word.into());
        let char_count = word.chars().count();
        let mut node = &mut self.root;

        for (idx, character) in word.chars().enumerate() {
            if !node.children.contains_key(&character) {
                let mut text = node.text.clone();
                text.push(character);
                let new_node = DagNode::new(text);
                node.children.insert(character, new_node);
            }
            node = node.children.get_mut(&character).unwrap();
            if idx == char_count - 1 {
                node.leaf = true;
                node.score = score;
                node.index = index;
            }
        }
    }

    fn decode_backward<'a>(&'a self, nodes: &'a Vec<Option<Node>>) -> Vec<&'a Node> {
        let mut next_node = nodes.last().unwrap();
        let mut best_sequence = vec![];

        while next_node.is_some() {
            let node_value = next_node.as_ref().unwrap();
            best_sequence.push(node_value);
            next_node = &nodes[node_value.start];
        }
        best_sequence.reverse();
        best_sequence
    }

    fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<&DagNode> {
        let mut results = vec![];
        let mut characters = text.chars();

        let mut node = self.root.children.get(&characters.next().unwrap());
        if node.is_some() {
            if node.unwrap().leaf {
                results.push(node.unwrap());
            }
        } else {
            return vec![];
        }
        while let Some(character) = characters.next() {
            node = node.unwrap().children.get(&character);
            if node.is_some() {
                if node.unwrap().leaf {
                    results.push(node.unwrap());
                }
            } else {
                break;
            }
        }
        results
    }

    fn decode_forward_dag<'a>(&'a self, text: &'a str) -> Vec<Option<Node>> {
        let mut char_positions = text.char_indices().map(|(pos, _)| pos).collect::<Vec<_>>();
        char_positions.push(text.len());
        let mut results = vec![None; char_positions.len()];
        let mut scores = vec![std::f32::NEG_INFINITY; char_positions.len()];
        scores[0] = 0f32;

        for char_start in 0..char_positions.len() - 1 {
            let matches = self.common_prefix_search(&text[char_positions[char_start]..]);
            for node in matches {
                let local_score = scores[char_start] + node.score;
                let char_end = char_start + node.len;
                if local_score > scores[char_end] {
                    results[char_end] = Some(Node {
                        index: node.index,
                        start: char_start,
                    });
                    scores[char_end] = local_score;
                }
            }
            if scores[char_start + 1] <= std::f32::MIN {
                results[char_start + 1] = Some(Node {
                    index: 0,
                    start: char_start,
                });
                scores[char_start + 1] = 0f32;
            }
        }
        results
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    fn tokenize(&self, text: &str) -> Vec<usize> {
        let text = (String::from(" ") + &text.replace('\n', " "))
            .replace(' ', &PREFIXED_UNDERSCORE.to_string());
        let text = text.as_str();
        let output = self.decode_forward_dag(text);
        let decoded = self.decode_backward(&output);
        decoded.into_iter().map(|node| node.index).collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        let mut out = String::new();
        for s in tokens.iter().map(|k| self.vocab.get(*k).unwrap()) {
            out += s;
        }
        out.replace(PREFIXED_UNDERSCORE, " ")
    }
}
