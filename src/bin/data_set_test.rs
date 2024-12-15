use rust_enn::nim::dp::*;

fn main() {
    // Example usage:
    let nim_heaps = vec![3, 4, 5];
    let bases = get_bases(nim_heaps.clone());
    let dp_size = bases.last().unwrap() * (nim_heaps.last().unwrap() + 1);
    let mut dp = vec![(-1, -1, false); dp_size];

    // Populate the DP table
    dp_alg(vec![3, 4, 5], nim_heaps.clone(), &mut dp, &bases);

    // Generate the dataset
    let (train_data, test_data) = create_data_set(&dp, nim_heaps.clone(), 0.8);

    println!("Training set size: {}", train_data.len());
    println!("Test set size: {}", test_data.len());

    // Example: Print a few samples from the training set
    for (state, mv) in train_data.iter().take(5) {
        println!("State: {:?} -> Move: {:?}", state, mv);
    }
}