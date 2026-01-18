/// Experimental Algorithm
/// 
/// Uses a compression-based fitness function.
/// Goal: Minimize total size (Token Count * Token ID Size + Vocabulary Size)
///

use std::collections::HashMap;

pub struct ExperimentalAlgo {
    pub vocab: HashMap<String, usize>, // Token -> ID
    pub next_id: usize,
}

impl ExperimentalAlgo {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            next_id: 0,
        }
    }

    /// Calculates the "fitness" of the current vocabulary on the given text.
    /// Higher is better (Compression Ratio).
    pub fn calculate_fitness(&self, text: &str) -> f64 {
        let original_size_bytes = text.len();
        
        let tokens = self.tokenize(text);
        
        // Size Estimation:
        // 1. Token Stream: Number of tokens * 2 bytes (assuming u16 IDs for simplicity)
        let token_stream_size = tokens.len() * 2; 
        
        // 2. Vocabulary Overhead: Size of storing the dictionary
        // Rough estimate: Sum of chars in vocab + overhead per entry
        let vocab_size: usize = self.vocab.keys().map(|k| k.len() + 4).sum();

        let compressed_size = token_stream_size + vocab_size;

        if compressed_size == 0 { return 0.0; }
        
        // Compression Ratio (e.g., 2.5x smaller)
        original_size_bytes as f64 / compressed_size as f64
    }

    pub fn train(&mut self, data: &str) {
        println!("Training on data length: {}", data.len());
        
        // Placeholder: Add some dummy tokens to test fitness
        let common_words = ["the", "and", "ing", "tion", " "];
        for word in common_words {
            if !self.vocab.contains_key(word) {
                self.vocab.insert(word.to_string(), self.next_id);
                self.next_id += 1;
            }
        }

        let fitness = self.calculate_fitness(data);
        println!("Initial Fitness (Compression Ratio): {:.4}x", fitness);
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple greedy tokenization for fitness calculation
        // In reality, you'd use a trie or max-match
        let mut tokens = Vec::new();
        let mut cursor = 0;
        let sorted_vocab: Vec<(String, usize)> = {
            let mut v: Vec<_> = self.vocab.clone().into_iter().collect();
            v.sort_by(|a, b| b.0.len().cmp(&a.0.len())); // Try longest first
            v
        };

        while cursor < text.len() {
            let mut match_found = false;
            let remaining = &text[cursor..];

            for (token_str, _) in &sorted_vocab {
                if remaining.starts_with(token_str) {
                    tokens.push(token_str.clone());
                    cursor += token_str.len();
                    match_found = true;
                    break;
                }
            }

            if !match_found {
                // If no token matches, take 1 char
                if let Some(ch) = remaining.chars().next() {
                    tokens.push(ch.to_string());
                    cursor += ch.len_utf8();
                } else {
                    break; 
                }
            }
        }
        tokens
    }
}
