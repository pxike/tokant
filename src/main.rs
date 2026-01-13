use dashmap::DashMap;
use rand::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const Q: f64 = 1.0; // Reduced to flatten hierarchy (less exponential reward for length)
const INITIAL_PHEROMONE: f64 = 1.0; 
const ALPHA: f64 = 1.0; 
const BETA: f64 = 2.0;  
const MAX_TOKEN_LEN: usize = 10; 

struct AntColony<'a> {
    // Shared pheromone map: Token -> Strength
    // DashMap for thread-safe concurrent access.
    pheromones: DashMap<&'a str, f64>, 
}

impl<'a> AntColony<'a> {
    fn new() -> Self {
        Self {
            pheromones: DashMap::new(),
        }
    }

    fn get_pheromone(&self, token: &str) -> f64 {
        // If not present, default to INITIAL_PHEROMONE
        match self.pheromones.get(token) {
            Some(val) => *val,
            None => INITIAL_PHEROMONE,
        }
    }

    // Heuristic: Prefer longer tokens. 
    // This drives the ants to discover chunks rather than single chars.
    fn get_heuristic(&self, token: &str) -> f64 {
        (token.len() as f64).powf(BETA)
    }

    /// Run a single ant on a slice of text (e.g., a line).
    /// Returns the list of tokens chosen and the number of steps.
    fn traverse(&self, text: &'a str, rng: &mut ThreadRng) -> (Vec<&'a str>, usize) {
        let mut tokens = Vec::with_capacity(text.len() / 4);
        let mut cursor = 0;
        let len = text.len();

        while cursor < len {
            // Identify candidates
            // We must use char_indices to ensure we slice at valid UTF-8 boundaries.
            let remaining = &text[cursor..];
            
            // Collect valid slices up to MAX_TOKEN_LEN characters long
            // Note: max_len logic is now based on chars, not bytes, which matches user intuition better.
            let mut candidates = Vec::with_capacity(MAX_TOKEN_LEN);
            let mut total_prob = 0.0;

            for (byte_offset, ch) in remaining.char_indices().take(MAX_TOKEN_LEN) {
                let end = byte_offset + ch.len_utf8();
                let token_slice = &remaining[..end]; // Valid UTF-8 slice
                
                let tau = self.get_pheromone(token_slice);
                let eta = self.get_heuristic(token_slice);
                
                let prob = tau.ln().max(0.0001) * eta;
                total_prob += prob;
                candidates.push((token_slice, prob));
            }

            // Selection
            // Default check if something went wrong (shouldn't if text is valid)
            if candidates.is_empty() {
                break;
            }

            let mut selected_token = candidates[0].0; 
            
            if total_prob > 0.0 {
                let threshold = rng.gen::<f64>() * total_prob;
                let mut current_sum = 0.0;
                
                for (tok, p) in candidates {
                    current_sum += p;
                    if current_sum >= threshold {
                        selected_token = tok;
                        break;
                    }
                }
            }

            tokens.push(selected_token);
            cursor += selected_token.len();
        }

        let steps = tokens.len();
        (tokens, steps)
    }

    /// Genetic Algorithm Selection:
    /// Keep Top 20% by pruning the bottom 80%.
    fn natural_selection(&self) {
        if self.pheromones.is_empty() { return; }

        let active_tokens = self.pheromones.len();
        
        // 1. Estimate Pruning Threshold via Sampling
        // Sorting 6 million items is slow. We can just purge anything below a moving average
        // or just strict top-k. Let's stick to the user's sort for now but optimize logic.
        
        let mut scores: Vec<f64> = self.pheromones.iter().map(|r| *r.value()).collect();
        // Sort descending
        scores.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let keep_ratio = 0.20;
        let cut_index = ((active_tokens as f64 * keep_ratio) as usize).max(1);
        let threshold = scores[cut_index.min(active_tokens - 1)];

        println!("  Natural Selection: Keeping top {:.0}% (Threshold: {:.4}). Active Tokens: {}", 
            keep_ratio * 100.0, threshold, active_tokens);

        // 2. Prune Weak Links
        self.pheromones.retain(|_, v| *v >= threshold);
    }

    /// Deposit pheromones using Logistic Growth (Soft Cap)
    /// This allows new tokens to grow fast, but prevents established ones from becoming infinite.
    fn deposit(&self, path: &[&'a str], _steps: usize) {
        const MAX_SCORE: f64 = 100000000.0; // The Carrying Capacity (K)

        for token in path {
            let len = token.len();
            if len > 1 {
                let reward = ((len - 1) as f64).powf(Q);
                
                // DashMap allows atomic updates via entry API
                let mut entry = self.pheromones.entry(token).or_insert(INITIAL_PHEROMONE);
                let current_val = *entry;
                
                // Logistic Update: dP = Reward * (1 - P/K)
                // If P is small, dP ≈ Reward (Fast Growth)
                // If P -> K, dP -> 0 (Saturation)
                let delta = reward * (1.0 - current_val / MAX_SCORE);
                
                if delta > 0.0 {
                    *entry += delta;
                }
            }
        }
    }
}

fn main() {
    // 1. Load Data
    let file_path = "text8"; // The 100MB corpus
    println!("Loading {}...", file_path);
    
    // Read file. Unwraps are for simplicity in this example.
    let raw_text = match fs::read_to_string(file_path) {
        Ok(t) => t,
        Err(_) => {
            println!("'text8' not found, falling back to 'sherlock.txt'");
            fs::read_to_string("sherlock.txt").unwrap_or_else(|_| "sample text".repeat(100))
        }
    };
    
    // Chunking Logic for Parallelism
    // If the file is 1 huge line (like text8), lines() gives 1 item => No parallelism.
    // We break it into fixed-size chunks (e.g., 500 chars) ensuring valid UTF-8.
    let chunk_size = 500;
    
    // Collecting char indices is memory intensive for 100MB (vector of tuples).
    // Let's use a smarter iterator approach or just byte slicing if ASCII (text8 is ASCII).
    // Text8 is pure ASCII lower case a-z and space. Byte slicing is safe.
    let lines: Vec<&str> = if raw_text.lines().count() < 1000 {
        println!("Detected monolithic text. Chunking into {}-byte segments...", chunk_size);
        raw_text.as_bytes()
            .chunks(chunk_size)
            .map(|c| std::str::from_utf8(c).unwrap_or(""))
            .collect()
    } else {
        println!("Detected structured text (lines). Using native newlines.");
        raw_text.lines()
            .filter(|l| !l.trim().is_empty())
            .collect()
    };

    let mut colony = AntColony::new();
    println!("Loaded {} lines. Starting training...", lines.len());

    let iterations = 20; 
    let start_time = Instant::now();

    for gen in 1..=iterations {
        let gen_start = Instant::now();
        
        // Parallel Traversal
        let results: Vec<(Vec<&str>, usize)> = lines.par_iter()
            .map(|&line| {
                let mut rng = rand::rng();
                colony.traverse(line, &mut rng)
            })
            .collect();

        // Deposit Pheromones
        results.into_par_iter().for_each(|(path, steps)| {
            colony.deposit(&path, steps);
        });

        // Global Selection (Survival of the Fittest)
        colony.natural_selection();

        let elapsed = gen_start.elapsed();
        let vocab_size = colony.pheromones.len();
        println!("Gen {}: Processed {} lines in {:.2?}. Vocab Size: {}", 
            gen,  lines.len(), elapsed, vocab_size);
    }

    println!("Training complete in {:.2?}.", start_time.elapsed());
    
    // Demo
    println!("\n--- Tokenization Demo ---");
    let sample = "ant colony optimization works well";
    let mut rng = rand::rng();
    let (tokens, _) = colony.traverse(sample, &mut rng);
    println!("Tokenized: {:?}", tokens);
    
    // Top Tokens
    let mut all_tokens: Vec<_> = colony.pheromones.iter()
        .map(|r| (r.key().to_string(), *r.value()))
        .collect();
    all_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    println!("\nTop 10 tokens:");
    for (t, s) in all_tokens.iter().take(10) {
        println!("'{}': {:.2}", t, s);
    }

    // Save full vocabulary
    // Filter out garbage (low score tokens that didn't repeat enough)
    let final_vocab: Vec<_> = all_tokens.into_iter()
        .collect();

    println!("\nSaving {} tokens to 'tokens.txt' (Pruned from {})...", final_vocab.len(), colony.pheromones.len());
    let mut file = std::io::BufWriter::new(fs::File::create("tokens.txt").expect("create failed"));
    use std::io::Write;
    
    // Write Header
    writeln!(file, "Token\tScore").ok();
    
    for (t, s) in final_vocab {
        // Escape newlines so the file remains line-based
        let safe_t = t.replace("\r", "\\r").replace("\n", "\\n");
        if let Err(_) = writeln!(file, "{}\t{:.4}", safe_t, s) {
            break; 
        }
    }
    println!("Done.");
}
