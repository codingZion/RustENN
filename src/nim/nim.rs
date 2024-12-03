use crate::neat::agent::Agent;
use crate::neat::population::GameResLog;
use crate::nim::dp;

#[derive(Clone)]
pub struct Nim {
    pub initial_state: Vec<u32>,
    pub input_size: usize,
    pub output_size: usize,
    pub dp: Vec<(i32, isize, bool)>,
    pub bases: Vec<usize>,
}

impl Nim {
    pub fn new(initial_state: Vec<u32>) -> Nim {
        let size = initial_state.len();
        let output_size = size + *initial_state.iter().max().unwrap() as usize;
        let nim_heaps:Vec<usize> = initial_state.clone().iter().map(|&x| x as usize).collect();
        let states_count = nim_heaps.iter().map(|&x| x + 1).product();
        println!("{}", states_count);
        let mut dp = vec![(-1, -1, false); states_count];
        dp[0] = (0, 0, true);
        dp::dp_alg(nim_heaps.clone(), nim_heaps.clone(), &mut dp, &dp::get_bases(nim_heaps.clone()));
        Nim {
            initial_state,
            input_size: size,
            output_size,
            dp,
            bases: dp::get_bases(nim_heaps.clone()),
        }
    }
    
    pub fn run_nim(&self, agents: Vec<&Agent>, print_game: bool, obj_eval: bool) -> GameResLog {
        let mut state = self.initial_state.clone();
        let mut turn = 0;
        let mut history = Vec::new();
        let mut perfect_play = vec![0; agents.len()];
        let mut turns_vec = vec![0; agents.len()];
        while state.iter().sum::<u32>() > 0 {
            //println!("state: {:?}, turn: {}", state, turn);
            let input = Self::get_input(state.clone());
            let agent = agents[turn % agents.len()];
            let output = agent.nn.predict(input);
            let agent_move = Self::get_output(output, state.clone());
            //println!("agent_move: {:?}", agent_move);
            if print_game {
                //println!("turn: {}, agent: {}", turn, turn % agents.len());
                //println!("state: {:?}, agent_move: {:?}", state, agent_move);
                history.push((state.clone(), agent_move));
            }
            turns_vec[turn % agents.len()] += 1;
            let old_state = state.clone();
            state[agent_move[0] as usize] -= agent_move[1] as u32;
            if (obj_eval) { perfect_play[turn % agents.len()] += self.is_perfect_play(old_state.clone(), state.clone()) as u32; }
            turn += 1;
            if turn > 500 {
                //println!("turns exceeded 500!");
                return (vec![0; agents.len()], history, turns_vec, perfect_play);
            }
        }
        let winner = (turn - 2) % agents.len();
        let mut res = vec![0; agents.len()];
        res[winner] = 1;
        (res, history, turns_vec, perfect_play)
    }
    
    pub fn run_nim_strict_state(&self, agents: Vec<&Agent>, print_game: bool, obj_eval: bool, state: &mut Vec<u32>) -> GameResLog {
        let mut turn = 0;
        let mut history = Vec::new();
        let mut perfect_play = vec![0; agents.len()];
        let mut turns_vec = vec![0; agents.len()]; 
        while state.iter().sum::<u32>() > 0 {
            //println!("state: {:?}, turn: {}", state, turn);
            let input = Self::get_input(state.clone());
            let agent = agents[turn % agents.len()];
            let output = agent.nn.predict(input);
            let agent_move = Self::get_output_raw(output, state.clone());
            //println!("agent_move: {:?}", agent_move);
            if print_game {
                //println!("turn: {}, agent: {}", turn, turn % agents.len());
                //println!("state: {:?}, agent_move: {:?}", state, agent_move);
                history.push((state.clone(), agent_move));
            }
            turn += 1;
            if state[agent_move[0] as usize] < agent_move[1] as u32 {
                if print_game {
                    //println!("invalid move!");
                }
                if obj_eval && state.iter().sum::<u32>() > 1 {
                    turns_vec[turn % agents.len()] += 1;
                }
                break;
            }
            if obj_eval { turns_vec[turn % agents.len()] += 1; }
            let old_state = state.clone();
            state[agent_move[0] as usize] -= agent_move[1] as u32;
            if obj_eval { perfect_play[turn % agents.len()] += self.is_perfect_play(old_state, state.clone()) as u32; }
            if turn > 500 {
                //println!("turns exceeded 500!");
                return (vec![0; agents.len()], history, turns_vec, perfect_play);
            }
        }
        let winner = (turn as isize - 2).abs() as usize % agents.len();
        let mut res = vec![0; agents.len()];
        res[winner] = 1;
        (res, history, turns_vec, perfect_play)
    }

    pub fn run_nim_strict(&self, agents: Vec<&Agent>, print_game: bool, obj_eval: bool) -> GameResLog {
        self.run_nim_strict_state(agents, print_game, obj_eval, &mut self.initial_state.clone())
    }

    pub fn run_nim_strict_random(&self, agents: Vec<&Agent>, print_game: bool, obj_eval: bool) -> GameResLog {
        let mut state = self.initial_state.clone();
        let max_states = state.clone();
        for i in state.iter_mut() {
            if max_states.iter().sum::<u32>() > 0 {
                *i = rand::random::<u32>() % *i + 1;
            }
        }
        self.run_nim_strict_state(agents, print_game, obj_eval, &mut state)
    }
    
    pub fn run_nim_strict_single(&self, agents: Vec<&Agent>, print_game: bool, obj_eval: bool) -> GameResLog {
        let mut state = vec![0; self.initial_state.len()];
        let i = rand::random::<u32>() as usize % state.len();
        state[i] = rand::random::<u32>() % self.initial_state[i] + 1;
        self.run_nim_strict_state(agents, print_game, obj_eval, &mut state)
    }
    
    pub fn is_perfect_play(&self, old_state: Vec<u32>, new_state: Vec<u32>) -> bool {
        if self.is_winning_state(old_state.clone()) {
            !self.is_winning_state(new_state.clone())
        }
        else {
            self.dp[dp::get_index(new_state.clone().iter().map(|&x| x as usize).collect(), &self.bases)].0 
                == self.dp[self.dp[dp::get_index(old_state.clone().iter().map(|&x| x as usize).collect(), &self.bases)].1 as usize].0
        }
    }
    
    pub fn is_winning_state(&self, state: Vec<u32>) -> bool {
        self.dp[dp::get_index(state.clone().iter().map(|&x| x as usize).collect(), &self.bases)].2
    }
    
    
    fn get_input(state: Vec<u32>) -> Vec<f64> {
        let mut input = Vec::new();
        for i in state.iter() {
            input.push(*i as f64);
        }
        input
    }
    
    fn get_output(output: Vec<f64>, state: Vec<u32>) -> [isize; 2] {
        let size = state.len();
        let mut res = [0, size as isize];
        for i in 0..size {
            if (output[i] > output[res[0] as usize] && state[i] > 0) || state[res[0] as usize] == 0 {
                res[0] = i as isize;
            }
        }
        for i in size..output.len() {
            if output[i] > output[res[1] as usize] {
                res[1] = i as isize;
            }
        }
        res[1] -= size as isize - 1;
        res[1] = res[1].min(state[res[0] as usize] as isize);
        res
    }

    fn get_output_raw(output: Vec<f64>, state: Vec<u32>) -> [isize; 2] {
        let size = state.len();
        let mut res = [0, size as isize];
        for i in 0..size {
            if output[i] > output[res[0] as usize]  {
                res[0] = i as isize;
            }
        }
        for i in size..output.len() {
            if output[i] > output[res[1] as usize] {
                res[1] = i as isize;
            }
        }
        res[1] -= size as isize - 1;
        res
    }
}