use dashmap::DashMap;
use rand::prelude::*;

const Q: f64 = 3.0; // Heavily reward length (Length^3) to beat frequency
const INITIAL_PHEROMONE: f64 = 1.0; 

const BETA: f64 = 4.0;  // Strong bias towards length exploration
const MAX_TOKEN_LEN: usize = 10; // Increase max length to find longer words

pub struct AntColony<'a> {
    // Shared pheromone map: Token -> Strength
    // DashMap for thread-safe concurrent access.
    pub pheromones: DashMap<&'a str, f64>, 
}

impl<'a> AntColony<'a> {
    pub fn new() -> Self {
        Self {
            pheromones: DashMap::new(),
        }
    }

    pub fn get_pheromone(&self, token: &str) -> f64 {
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
    pub fn traverse(&self, text: &'a str, rng: &mut ThreadRng) -> (Vec<&'a str>, usize) {
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
                let threshold = rng.random::<f64>() * total_prob;
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
    pub fn natural_selection(&self) {
        if self.pheromones.is_empty() { return; }

        let active_tokens = self.pheromones.len();
        
        // 1. Estimate Pruning Threshold via Sampling
        // Sorting 6 million items is slow. We can just purge anything below a moving average
        // or just strict top-k. Let's stick to the user's sort for now but optimize logic.
        
        let mut scores: Vec<f64> = self.pheromones.iter().map(|r| *r.value()).collect();
        // Sort descending
        scores.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let keep_ratio = 0.50; // Relaxed selection (50%) to give long tokens time to survive
        let cut_index = ((active_tokens as f64 * keep_ratio) as usize).max(1);
        let threshold = scores[cut_index.min(active_tokens - 1)];

        println!("  Natural Selection: Keeping top {:.0}% (Threshold: {:.4}). Active Tokens: {}", 
            keep_ratio * 100.0, threshold, active_tokens);

        // 2. Prune Weak Links & Evaporate Survivors
        // Trim 20% of every score (keeping them hungry)
        self.pheromones.retain(|_, v| {
            if *v < threshold {
                return false; // Eliminate
            }
            *v *= 0.8;
            true
        });
    }

    /// Deposit pheromones using Logistic Growth (Soft Cap)
    /// This allows new tokens to grow fast, but prevents established ones from becoming infinite.
    pub fn deposit(&self, path: &[&'a str], _steps: usize) {
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
