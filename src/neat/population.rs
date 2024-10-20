use std::fs::OpenOptions;
use std::io::Write;
use std::fmt::Write as FmtWrite;
use std::ops::Add;
use crate::neat::agent::Agent;
use std::sync::{Arc, Mutex};
use std::thread;
use indicatif::ProgressBar;
// import csv library
use csv;
use serde::{Deserialize, Serialize};
use std::thread::JoinHandle;
use rayon::prelude::*; // For parallel iterators
use rand::Rng;
use std::string::String;
// For random number generation

pub type GameResLog = (Vec<u32>, Vec<(Vec<u32>, [usize; 2])>, Vec<u32>, Vec<u32>);

#[derive(Clone, Serialize, Deserialize)]
pub struct Population<T: Send + Sync + 'static> {
    pub size: u32,
    pub cycle: u32,
    pub agents: Vec<Agent>,
    pub inputs: usize,
    pub outputs: usize,
    // serde ignore
    #[serde(skip, default = "default_fn")]
    pub run_game: fn(&T, Vec<&Agent>, bool, bool) -> GameResLog,  // Function pointer in the Population struct
    pub mutation_rate_range: (usize, usize),
    pub best_agents: Vec<Agent>,
    pub best_agents_comp_res: Vec<u32>,
    pub best_agents_won_games: u32,
}

fn default_fn<T: Send + Sync + 'static>() -> fn(&T, Vec<&Agent>, bool, bool) -> GameResLog {
    default_game
}
fn default_game<T: Send + Sync + 'static>(_: &T, _: Vec<&Agent>, _: bool, _: bool) -> GameResLog {
    (vec![], vec![], vec![], vec![])
}

impl<T: Send + Sync + 'static + Clone> Population<T> {
    pub fn new(size: u32, inputs: usize, outputs: usize, run_game: fn(&T, Vec<&Agent>, bool, bool) -> GameResLog, mutation_rate_range: (usize, usize)) -> Population<T> {
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
            best_agents: Vec::new(),
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
        let bar = ProgressBar::new(games as u64);

        let step = (self.size as usize / games) + 1;
        let offset = rand::thread_rng().gen_range(0..step); // Optimized random offset generation

        // Precompute pairings outside the parallel loop to avoid mutating `self`
        let pairings: Vec<_> = (0..games)
            .map(|i| self.circular_pairing(i * step + offset))
            .collect();

        // Mutex for safe concurrent updates of agent fitness
        let fitness_updates = Arc::new(Mutex::new(vec![0.0; self.agents.len()]));

        // Use `rayon`'s thread pool to run games in parallel
        pairings.into_par_iter().for_each(|pairing| {
            bar.inc(1);
            for j in pairing {
                let agents = Arc::clone(&agents); // Only clone if needed
                let game = Arc::clone(&game_arc); // Only clone if needed
                let fitness_updates = Arc::clone(&fitness_updates); // Clone fitness mutex

                // Run the game and get the results
                let agents_slice = vec![&agents[j[0]], &agents[j[1]]];
                let game_res = run_game(&game, agents_slice, false, false).0;

                // Update fitness in a critical section
                let mut fitness_lock = fitness_updates.lock().unwrap();
                fitness_lock[j[0]] += game_res[0] as f64;
                fitness_lock[j[1]] += game_res[1] as f64;
            }
        });

        // After all threads finish, update the agents' fitness from the results
        let fitness_lock = fitness_updates.lock().unwrap();
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.fitness = fitness_lock[i];
        }
    }
    
    //compete_best_agents() but multithreaded
   pub fn compete_best_agents_mt(&mut self, game: &T, agent: &Agent) -> (Vec<u32>, Vec<(usize, usize, GameResLog)>) {
        let game_arc = Arc::new(game.clone());
        let best_agents_arc = Arc::new(self.best_agents.clone()); // Clone `self.best_agents` into the `Arc`
        let print_games_mutex = Arc::new(Mutex::new(Vec::new()));

        if best_agents_arc.len() == 0 {
            return (vec![], vec![]);
        }

        let print_agent = rand::thread_rng().gen_range(0..best_agents_arc.len());
        let run_game = self.run_game;

        self.best_agents_comp_res = vec![0; best_agents_arc.len()];

        // Mutex to protect the shared results
        let results_mutex = Arc::new(Mutex::new(vec![0; best_agents_arc.len()]));

        // Prepare pairings upfront for easier parallel processing
        let pairings: Vec<_> = (0..best_agents_arc.len())
            .flat_map(|i| {
                let best_agents = Arc::clone(&best_agents_arc); // Clone the `Arc` to use inside the closure
                (0..2).map(move |j| {
                    // Dereference `best_agents` and clone the required agents from the underlying Vec
                    let mut agents = vec![agent.clone(), best_agents[i].clone()];
                    if j == 1 {
                        agents.reverse();
                    }
                    (i, j, agents)
                })
            })
            .collect();

        // Use `rayon` for parallel execution
        pairings.into_par_iter().for_each(|(i, j, agents)| {
            let game = Arc::clone(&game_arc);
            let results_mutex = Arc::clone(&results_mutex);
            let print_games_mutex = Arc::clone(&print_games_mutex);

            // Decide whether to print or not
            let game_res_log = if i == print_agent || i == best_agents_arc.len() - 1 {
                let agents_ref = vec![&agents[0], &agents[1]];
                let mut str = String::new();
                if j == 0 {
                    writeln!(&mut str, "best_agent vs agent: {}", i).unwrap();
                } else { 
                    writeln!(&mut str, "agent: {} vs best_agent", i).unwrap();
                }
                let res = run_game(&game, agents_ref, true, true);
                for i in 0..res.1.len() {
                    let (state, agent_move) = res.1[i].clone();
                    writeln!(&mut str, "Turn {}: state: {:?}, agent_move: {:?}", i, state, agent_move).unwrap();
                }
                writeln!(&mut str, "game_res: {:?}", res.0).unwrap();
                println!("{}", str);
                res
            } else {
                let agents_ref = vec![&agents[0], &agents[1]];
                run_game(&game, agents_ref, false, true)
            };
            
            let game_res = game_res_log.0;

            // Update the results safely
            let mut results_lock = results_mutex.lock().unwrap();
            results_lock[i] += if j == 0 { game_res[0] } else { game_res[1] };
            if i == print_agent || i == best_agents_arc.len() - 1 {
                let mut print_games_lock = print_games_mutex.lock().unwrap();
                if j == 0 {
                    print_games_lock.push((self.cycle as usize, i, game_res_log.clone()));
                } else {
                    print_games_lock.push((i, self.cycle as usize, game_res_log.clone()));
                }
            }
        });

        // Retrieve the final results from the mutex
        let final_results = results_mutex.lock().unwrap().clone();
        self.best_agents_comp_res = final_results.clone();
        self.best_agents_won_games = self.best_agents_comp_res.iter().sum();
        let print_games = print_games_mutex.lock().unwrap().clone();

        (final_results, print_games)
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
    /*
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
            wtr.write_record(&[self.best_agents_won_games.to_string(), (self.best_agents_won_games as f64 / self.best_agents_comp_res.len() as f64 * 50.0).to_string(), comp_res]).unwrap();
        }
        else {
            println!("Can't save best agent tournament csv, best_agents_comp_res is empty!");
        }
        Ok(())
    }
*/
    pub fn create_best_agent_games_txt(&self, filename: &str) {
        OpenOptions::new()
            .write(true)
            .create(true)
            .open(filename)
            .unwrap();
    }
    pub fn save_best_agent_games_txt(&self, filename: &str, opponent: usize, games: Vec<GameResLog>) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .unwrap();
        
        writeln!(file, "Generation: {}", self.cycle).unwrap();
        for i in 
        
        

    }

    // function that saves itself to a bincode file with serde and bincode crate
    pub fn save_population(&self, filename: &str) -> JoinHandle<Result<(), Box<std::io::Error>>> {
        let filename = filename.to_string();
        let temp_file = filename.clone().add(".tmp");
        let population = self.clone();
        thread::spawn(move || {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .open(&temp_file)
                .unwrap();
            bincode::serialize_into(file, &population).unwrap();
            std::fs::rename(temp_file, filename).unwrap();
            Ok(())
        })
    }

    pub fn load_population(filename: &str) -> Result<Population<T>, Box<std::io::Error>> {
        let file = OpenOptions::new()
            .read(true)
            .open(filename)
            .unwrap();
        let res: Population<T> = bincode::deserialize_from(file).unwrap();
        Ok(res)
    }
}
