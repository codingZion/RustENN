use rust_enn::enn::agent::NeuralNetwork;
fn main() {
    let nn = NeuralNetwork::new(5, 12);
    println!("{:?}", nn);
}