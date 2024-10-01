use std::time::Instant;
use std::thread::available_parallelism;

fn main() {
    println!("Available parallelism: {}", available_parallelism().unwrap().get());
    let test_size = 1000000000;
    let start = Instant::now();
    let mut a = 0;
    for _ in 0..test_size {
        a += 1;
    }
    println!("Test 1: {:?}", start.elapsed());
    println!("a: {}", a);
    let start = Instant::now();
    let mut c = Vec::new(); 
    for i in 0..test_size {
        c.push(i);
    }
    println!("Test 2: {:?} ms", start.elapsed());
    let start = Instant::now();
    let mut b = 0.5;
    for _ in 0..test_size {
        b *= 1. - b;
    }
    println!("b: {}", b);
    println!("Test 3: {:?}", start.elapsed());
}