use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::env;

struct Tokenizer {
    vocab: HashMap<String, f64>,
    max_token_len: usize,
}

impl Tokenizer {
    fn load(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut vocab = HashMap::new();
        let mut max_len = 0;

        for line in io::BufReader::new(file).lines() {
            let line = line?;
            if line.starts_with("Token\t") { continue; } // Header
            
            if let Some((token, score_str)) = line.rsplit_once('\t') {
                // Determine max len (in chars) for optimization
                let char_count = token.chars().count();
                if char_count > max_len {
                    max_len = char_count;
                }
                
                // Parse Score
                let score = score_str.trim().parse::<f64>().unwrap_or(0.0);

                // We assume tokens.txt has escaped chars like \n, \r
                // We need to unescape them to match correctly against real text
                let unescaped = token.replace("\\n", "\n").replace("\\r", "\r");
                vocab.insert(unescaped, score); 
            }
        }
        
        Ok(Self { vocab, max_token_len: max_len })
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut cursor = 0; // Index in the 'chars' vector, not byte offset
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let char_count = chars.len();

        while cursor < char_count {
            let mut best_match: Option<String> = None;
            let mut best_len = 0;
            let mut best_score = -1.0;

            // Greedy Search: Try ALL possible slices, pick highest Score
            // Start from min(remaining_chars, max_token_len)
            let remaining = char_count - cursor;
            let search_depth = std::cmp::min(remaining, self.max_token_len);
            
            // Iterate length j from 1 to search_depth
            for j in 1..=search_depth {
                 let start_byte = chars[cursor].0;
                 
                 // Calculate end byte
                 let end_index = cursor + j;
                 let end_byte = if end_index < char_count { 
                     chars[end_index].0 
                 } else { 
                     text.len() 
                 };
                 
                 let slice = &text[start_byte..end_byte];
                 
                 if let Some(&score) = self.vocab.get(slice) {
                     // Check if this token is "better" than what we found so far
                     // Priority: Higher Score > Longer Length
                     if score > best_score || (score == best_score && j > best_len) {
                        best_match = Some(slice.to_string());
                        best_len = j; 
                        best_score = score;
                     }
                 }
            }

            if let Some(token) = best_match {
                tokens.push(token);
                cursor += best_len;
            } else {
                // If no match found, consume one character
                let start_byte = chars[cursor].0;
                let end_index = cursor + 1;
                let end_byte = if end_index < char_count { chars[end_index].0 } else { text.len() };
                
                let char_slice = &text[start_byte..end_byte];
                tokens.push(char_slice.to_string());
                cursor += 1;
            }
        }
        tokens
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Default demo text if no args provided
    let default_text = "the quick brown fox jumps over the lazy dog two zero zero nine";
    let input_text = if args.len() > 1 { &args[1] } else { default_text };

    match Tokenizer::load("tokens.txt") {
        Ok(tokenizer) => {
            println!("Loaded vocab size: {} tokens", tokenizer.vocab.len());
            println!("Max token length in vocab: {} chars", tokenizer.max_token_len);
            println!("\nInput Text:\n'{}'", input_text);
            
            let start = std::time::Instant::now();
            let result = tokenizer.tokenize(input_text);
            let elapsed = start.elapsed();
            
            println!("\nTokenized Output ({} tokens found in {:.2?}):", result.len(), elapsed);
            println!("{:?}", result);
            
            // Pretty print reconstruction check
            println!("\nReconstructed: '{}'", result.join(""));
        }
        Err(e) => eprintln!("Failed to load 'tokens.txt'. Make sure you ran the training first.\nError: {}", e),
    }
}
