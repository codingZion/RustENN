pub struct GameState<T>{
    pub board: T,
    pub winner: Option<usize>,
    pub turn: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub input: Vec<f64>,
    pub output: Vec<f64>,
}

