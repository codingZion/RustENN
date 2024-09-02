use std::fs::OpenOptions;
use crate::neat::agent::Agent;
use std::sync::{Arc, Mutex};
use std::thread;
use indicatif::ProgressBar;
// import csv library
use csv;
use csv::Error;

macro_rules! noop { () => (); }

pub struct Population<T: Send + Sync + 'static> {
    pub size: u32,
    pub cycle: u32,
    pub agents: Vec<Agent>,
    pub inputs: usize,
    pub outputs: usize,
    pub run_game: fn(&T, Vec<&Agent>, bool) -> Vec<u32>,  // Function pointer in the Population struct
    pub mutation_rate_range: (usize, usize),
    pub best_agents: Arc<Mutex<Vec<Agent>>>,
    pub best_agents_comp_res: Vec<u32>,
    pub best_agents_won_games: u32,
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
            agents,
            inputs,
            outputs,
            run_game,
            mutation_rate_range,
            best_agents: Arc::new(Mutex::new(Vec::new())),
            best_agents_comp_res: Vec::new(),
            best_agents_won_games: 0,
        }
    }

    pub fn rank_agents(&mut self) {
        let mut agents = &mut self.agents;
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
        let game_arc = Arc::new(game.clone());
        // Reset fitness for all agents
        for agent in self.agents.iter_mut() {
            agent.fitness = 0.0;
        }

        let agents = Arc::new(self.agents.clone());
        let run_game = self.run_game; // Capture the run_game function pointer from the struct

        let mut handles = vec![]; // Vector to store thread handles

        let step = (self.size as usize / games) + 1;
        let offset = rand::random::<u32>() as usize % (step);
        let bar = ProgressBar::new(games as u64);
        for i in 0..games {
            bar.inc(1);
            for j in self.circular_pairing(i * step + offset) {
                let agents = Arc::clone(&agents);
                let game = Arc::clone(&game_arc);
                let handle = thread::spawn(move || {
                    let agents_slice = vec![&agents[j[0]], &agents[j[1]]];
                    let game_res = run_game(&game, agents_slice, false);  // Use the run_game function pointer
                    [(j[0], game_res[0]), (j[1], game_res[1])]
                });
                handles.push(handle); // Store the handle
                while handles.len() >= 6 {
                    let mut i = 0;
                    while i < handles.len() {
                        if handles[i].is_finished() {
                            let res = handles.remove(i).join().unwrap();
                            self.agents[res[0].0].fitness += res[0].1 as f64;
                            self.agents[res[1].0].fitness += res[1].1 as f64;
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }

        // Wait for all threads to finish
        for handle in handles {
            let res = handle.join().unwrap();
            self.agents[res[0].0].fitness += res[0].1 as f64;
            self.agents[res[1].0].fitness += res[1].1 as f64;
        }
    }

    pub fn compete_best_agents(&mut self, game: &T, agent: &Agent) -> Vec<u32> {
        let best_agents = self.best_agents.lock().unwrap();
        let mut res = vec![0; best_agents.len()];
        let print_agent = rand::random::<u32>() % (if best_agents.len() > 0 {best_agents.len()} else {1}) as u32;
        for i in 0..best_agents.len() {
            let mut agents = Vec::new();
            agents.push(agent);
            agents.push(&best_agents[i]);
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
            agents.push(&best_agents[i]);
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
        self.best_agents_comp_res = res.clone();
        self.best_agents_won_games = res.iter().sum();
        res
    }
    
    //compete_best_agents() but multithreaded
    pub fn compete_best_agents_mt(&mut self, game: &T, agent: &Agent) -> Vec<u32> {
        let game = Arc::new(game.clone());
        let agent = Arc::new(agent.clone());
        let best_agents = Arc::clone(&self.best_agents);
        let res = Arc::new(Mutex::new(vec![0; best_agents.lock().unwrap().len()]));
        let print_agent = rand::random::<u32>() % (if best_agents.lock().unwrap().len() > 0 {best_agents.lock().unwrap().len()} else {1}) as u32;
        let run_game = self.run_game;
        let mut handles = vec![];

        if best_agents.lock().unwrap().len() == 0 {
            return vec![0; 0];
        }
        for i in 0..best_agents.lock().unwrap().len() {
            let game = Arc::clone(&game);
            let agent = Arc::clone(&agent);
            let res = Arc::clone(&res);
            let best_agents = Arc::clone(&best_agents);

            let handle = thread::spawn(move || {
                let best_agents_lock = best_agents.lock().unwrap();
                let mut agents = vec![&agent, &best_agents_lock[i]];

                let mut game_res;
                if i == print_agent as usize {
                    println!("best_agent vs agent: {}", i);
                    game_res = run_game(&game, agents, true);
                    println!("game_res: {:?}", game_res);
                } else {
                    game_res = run_game(&game, agents, false);
                }
                let mut res_lock = res.lock().unwrap();
                res_lock[i] = game_res[0];

                let mut agents = vec![&best_agents_lock[i], &agent];

                if i == print_agent as usize {
                    println!("agent: {} vs best_agent", i);
                    game_res = run_game(&game, agents, true);
                    println!("game_res: {:?}", game_res);
                } else {
                    game_res = run_game(&game, agents, false);
                }
                res_lock[i] += game_res[1];

            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        self.best_agents_comp_res = res.lock().unwrap().clone();
        self.best_agents_won_games = self.best_agents_comp_res.iter().sum();

        // Return the result by unwrapping the Arc and Mutex
        Arc::try_unwrap(res).unwrap().into_inner().unwrap()
    }

    pub fn evolve(&mut self) {
        let mut new_agents = Vec::new();
        //add old best agents to new population
        let mut best_agents = self.best_agents.lock().unwrap().clone();
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
                let agents_lock = &self.agents;
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
        self.agents.clone_from(&new_agents);
        self.cycle += 1;
    }

    pub fn create_stats_csv(&self, filename: &str) -> Result<(), Box<csv::Error>> {
        let writer_result = csv::Writer::from_path(filename);
        let mut wtr = match writer_result {
            Ok(writer) => writer,
            Err(err) => return Err(Box::new(err)),
        };

        match wtr.write_record(&["fitness", "layer_sizes", "edge_count"]) {
            Ok(_) => Ok(()),
            Err(err) => return Err(Box::new(err)),
        }
    }

    pub fn save_stats_csv(&self, filename: &str) -> Result<(), Box<csv::Error>> {
        let file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .unwrap();
        let mut wtr = csv::Writer::from_writer(file);
        let agents = &self.agents;
        for agent in agents.iter() {
            wtr.write_record(&[agent.fitness.to_string(), agent.nn.layer_sizes.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","), agent.nn.edge_count.to_string()]).unwrap();
        }
        Ok(())
    }
    pub fn create_best_agent_tournament_csv(&self, filename: &str) -> Result<(), Box<csv::Error>> {
        let writer_result = csv::Writer::from_path(filename);
        let mut wtr = match writer_result {
            Ok(writer) => writer,
            Err(err) => return Err(Box::new(err)),
        };

        match wtr.write_record(&["won_games", "percentage", "competition_results"]) {
            Ok(_) => Ok(()),
            Err(err) => return Err(Box::new(err)),
        }
    }
    pub fn save_best_agent_tournament_csv(&self, filename: &str) -> Result<(), Box<csv::Error>> {
        let file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .unwrap();
        let mut wtr = csv::Writer::from_writer(file);


        let mut comp_res = String::new();
        if self.best_agents_comp_res.len() > 0 {
            for i in self.best_agents_comp_res.iter() {
                comp_res.push_str(&i.to_string());
            }
        }
        else {
            comp_res.push_str("No competition");
        }
        wtr.write_record(&[self.best_agents_won_games.to_string(), (self.best_agents_won_games as f64 / self.best_agents_comp_res.len() as f64 * 50.0).to_string(), comp_res]).unwrap();
        Ok(())
    }
}
