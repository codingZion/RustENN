use crate::neat::agent::Agent;
use std::sync::{Arc, Mutex};
use std::thread;
use indicatif::ProgressBar;

pub struct Population<T: Send + Sync + 'static> {
    pub size: u32,
    pub cycle: u32,
    pub agents: Arc<Mutex<Vec<Agent>>>,
    pub inputs: usize,
    pub outputs: usize,
    pub run_game: fn(&T, Vec<&Agent>, bool) -> Vec<u32>,  // Function pointer in the Population struct
    pub mutation_rate_range: (usize, usize),
    pub best_agents: Vec<Agent>,
}

impl<T: Send + Sync + 'static + Clone> Population<T> {
    pub fn new(size: u32, inputs: usize, outputs: usize, run_game: fn(&T, Vec<&Agent>, bool) -> Vec<u32>, mutation_rate_range: (usize, usize)) -> Population<T> {
        let mut agents = Vec::new();
        for _ in 0..size {
            agents.push(Agent::new(inputs, outputs));
        }
        Population {
            size,
            cycle: 0,
            agents: Arc::new(Mutex::new(agents)),
            inputs,
            outputs,
            run_game,
            mutation_rate_range,
            best_agents: Vec::new(),
        }
    }

    pub fn rank_agents(&mut self) {
        let mut agents = self.agents.lock().unwrap();
        agents.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        for i in 0..self.size as usize {
            agents[i].rank = i as isize;
        }
        /*println!("fitnesses: ");
        for i in 0..self.size as usize {
            print!("{}: {}, ", i, agents[i].fitness);
        }*/
    }
    
    pub fn circular_pairing(&mut self, distance: usize) -> Vec<[usize; 2]> {
        let mut res = Vec::new();
        for i in 0..self.size as usize {
            res.push([i , (i + distance) % self.size as usize]);
        }
        res
    }

    pub fn competition(&mut self, game: &T, games: usize) {
        // Reset fitness for all agents
        for agent in self.agents.lock().unwrap().iter_mut() {
            agent.fitness = 0.0;
        }

        let agents = Arc::clone(&self.agents);
        let run_game = self.run_game; // Capture the run_game function pointer from the struct

        let mut handles = vec![]; // Vector to store thread handles

        let step = (self.size as usize / games) + 1;
        let offset = rand::random::<u32>() as usize % (step);
        let bar = ProgressBar::new(games as u64);
        for i in 0..games {
            bar.inc(1);
            for j in self.circular_pairing(i * step + offset) {
                let agents = Arc::clone(&agents);
                let game = game.clone(); // Clone game for each thread
                let handle = thread::spawn(move || {
                    let mut agents_lock = agents.lock().unwrap();
                    let agents_slice = vec![&agents_lock[j[0]], &agents_lock[j[1]]];
                    let game_res = run_game(&game, agents_slice, false);  // Use the run_game function pointer
                    agents_lock[j[0]].fitness += game_res[0] as f64;
                    agents_lock[j[1]].fitness += game_res[1] as f64;
                });
                handles.push(handle); // Store the handle
            }
        }

        // Wait for all threads to finish
        for handle in handles {
            handle.join().unwrap();
        }
    }

    pub fn compete_best_agents(&mut self, game: &T, agent: &Agent) -> Vec<u32> {
        let mut res = vec![0; self.best_agents.len()];
        let print_agent = rand::random::<u32>() % (if self.best_agents.len() > 0 {self.best_agents.len()} else {1}) as u32;
        for i in 0..self.best_agents.len() {
            let mut agents = Vec::new();
            agents.push(agent);
            agents.push(&self.best_agents[i]);
            let game_res;
            if i == print_agent as usize {
                println!("best_agent vs agent: {}", i);
                game_res = (self.run_game)(game, agents, true);
                println!("game_res: {:?}", game_res);
            } else {
                game_res = (self.run_game)(game, agents, false);
            }
            res[i] = game_res[0];
            let mut agents = Vec::new();
            agents.push(&self.best_agents[i]);
            agents.push(agent);
            let game_res;
            if i == print_agent as usize {
                println!("agent: {} vs best_agent", i);
                game_res = (self.run_game)(game, agents, true);
                println!("game_res: {:?}", game_res);
            } else {
                game_res = (self.run_game)(game, agents, false);
            }
            res[i] += game_res[1];
        }
        res
    }

    pub fn evolve(&mut self) {
        let mut new_agents = Vec::new();
        //add old best agents to new population
        let mut best_agents = self.best_agents.clone();
        for _ in 0..self.size / 10 {
            if !best_agents.is_empty() {
                let i = rand::random::<u64>() as usize % best_agents.len();
                new_agents.push(best_agents.remove(i).clone());
            }
        }
    
    
        //add random agents to new population
        for _ in 0..self.size / 10 {
            new_agents.push(Agent::new(self.inputs, self.outputs));
        }
        let mut i = 0;
        while new_agents.len() < self.size as usize {
            for _ in 0..((self.size as usize - new_agents.len()) / 7).max(1) {
                let agents_lock = self.agents.lock().unwrap();
                new_agents.push(
                    agents_lock[i].clone().mutate(
                        rand::random::<u64>() as usize % (self.mutation_rate_range.1 - self.mutation_rate_range.0) + self.mutation_rate_range.0
                    )
                );
                if new_agents.len() >= self.size as usize {
                    break;
                }
            }
            i += 1;
        }
        self.agents.lock().unwrap().clone_from(&new_agents);
        self.cycle += 1;
    }
}
