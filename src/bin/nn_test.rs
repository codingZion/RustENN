use rust_neat::neat::agent::NeuralNetwork;
fn main() {
    let nn = NeuralNetwork::new(5, 12);
    println!("{:?}", nn);
}