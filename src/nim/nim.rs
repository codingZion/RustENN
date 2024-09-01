use crate::neat::agent::Agent;

#[derive(Clone)]
pub struct Nim {
    pub initial_state: Vec<u32>,
    pub input_size: usize,
    pub output_size: usize,
}

impl Nim {
    pub fn new(initial_state: Vec<u32>) -> Nim {
        let size = initial_state.len();
        let output_size = size + *initial_state.iter().max().unwrap() as usize;
        Nim {
            initial_state,
            input_size: size,
            output_size,
        }
    }
    
    pub fn run_nim(&self, agents: Vec<&Agent>, print_game: bool) -> Vec<u32> {
        let mut state = self.initial_state.clone();
        let mut turn = 0;
        while state.iter().sum::<u32>() > 0 {
            //println!("state: {:?}, turn: {}", state, turn);
            let input = Self::get_input(state.clone());
            let agent = agents[turn % agents.len()];
            let output = agent.nn.predict(input);
            let agent_move = Self::get_output(output, state.clone());
            //println!("agent_move: {:?}", agent_move);
            if print_game {
                println!("turn: {}, agent: {}", turn, turn % agents.len());
                println!("state: {:?}, agent_move: {:?}", state, agent_move);
            }
            state[agent_move[0]] -= agent_move[1] as u32;
            turn += 1;
            if turn > 500 {
                println!("turns exceeded 500!");
                return vec![0; agents.len()];
            }
        }
        let winner = (turn - 2) % agents.len();
        let mut res = vec![0; agents.len()];
        res[winner] = 1;
        res
    }

    pub fn run_nim_strict(&self, agents: Vec<&Agent>, print_game: bool) -> Vec<u32> {
        let mut state = self.initial_state.clone();
        let mut turn = 0;
        while state.iter().sum::<u32>() > 0 {
            //println!("state: {:?}, turn: {}", state, turn);
            let input = Self::get_input(state.clone());
            let agent = agents[turn % agents.len()];
            let output = agent.nn.predict(input);
            let agent_move = Self::get_output_raw(output, state.clone());
            //println!("agent_move: {:?}", agent_move);
            if print_game {
                println!("turn: {}, agent: {}", turn, turn % agents.len());
                println!("state: {:?}, agent_move: {:?}", state, agent_move);
            }
            turn += 1;
            if state[agent_move[0]] < agent_move[1] as u32 {
                if print_game {
                    println!("invalid move!");
                }
                break;
            }
            state[agent_move[0]] -= agent_move[1] as u32;
            if turn > 500 {
                println!("turns exceeded 500!");
                return vec![0; agents.len()];
            }
        }
        let winner = (turn - 2) % agents.len();
        let mut res = vec![0; agents.len()];
        res[winner] = 1;
        res
    }
    
    fn get_input(state: Vec<u32>) -> Vec<f64> {
        let mut input = Vec::new();
        for i in state.iter() {
            input.push(*i as f64);
        }
        input
    }
    
    fn get_output(output: Vec<f64>, state: Vec<u32>) -> [usize; 2] {
        let size = state.len();
        let mut res = [0, size];
        for i in 0..size {
            if (output[i] > output[res[0]] && state[i] > 0) || state[res[0]] == 0 {
                res[0] = i;
            }
        }
        for i in size..output.len() {
            if output[i] > output[res[1]] {
                res[1] = i - size + 1;
            }
        }
        res[1] = res[1].min(state[res[0]] as usize);
        res
    }
    
    fn get_output_raw(output: Vec<f64>, state: Vec<u32>) -> [usize; 2] {
        let size = state.len();
        let mut res = [0, size];
        for i in 0..size {
            if output[i] > output[res[0]]  {
                res[0] = i;
            }
        }
        for i in size..output.len() {
            if output[i] > output[res[1]] {
                res[1] = i - size + 1;
            }
        }
        res
    }
}