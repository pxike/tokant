
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

use tokant::algorithms::ant_colony::AntColony;




fn main() {
    // 1. Load Data
    let file_path = "AllCombined.txt"; // The 100MB corpus
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
    // We break it into fixed-size chunks (e.g., 1000 chars) ensuring valid UTF-8.
    let chunk_size = 1000;
    
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

    let colony = AntColony::new();
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
