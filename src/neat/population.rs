use std::fs::OpenOptions;
use std::io::Write;
use std::fmt::Write as FmtWrite;
use std::ops::Add;
use crate::neat::agent::{Agent, MUTATION_TYPES};
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
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

// For random number generation

pub type GameResLog = (Vec<u32>, Vec<(Vec<u32>, [usize; 2])>, Vec<u32>, Vec<u32>);

pub const MOVE_FITNESS: bool = false;

pub const FITNESS_EXP: f64 = 1.5;

pub const BEST_AGENT_TOURNAMENT_MAX: usize = 50;

pub const BEST_AGENT_SHARE: u32 = 15;
pub const RANDOM_AGENT_SHARE: u32 = 15;

pub const RANDOM_OLD_AGENT_SHARE: u32 = 15;


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
    pub comp_avg_moves: f64,
    pub best_agents: Vec<Agent>,
    pub best_agents_comp_res: Vec<u32>,
    pub best_agents_won_games: u32,
    pub best_agent_played_games: u32,
    pub best_agents_comp_moves: Vec<u32>,
    pub best_agents_comp_avg_moves: f64,
    pub best_agent_performances: Vec<u32>,
    pub best_agent_avg_performances: f64,
    pub total_elapsed: f64,
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
            comp_avg_moves: 0.,
            best_agents_comp_res: Vec::new(),
            best_agents_won_games: 0,
            best_agent_played_games: 0,
            best_agents_comp_moves: Vec::new(),
            best_agents_comp_avg_moves: 0.,
            best_agent_performances: Vec::new(),
            best_agent_avg_performances: 0.,
            total_elapsed: 0.,
        }
    }

    pub fn rank_agents(&mut self) {
        let agents = &mut self.agents;
        agents.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        for i in 0..self.size as usize {
            agents[i].rank = i as isize;
        }
        /*
        for i in agents {
            print!("{}, ", i.fitness);
        }
        println!();*/
    }

    pub fn circular_pairing(&mut self, distance: usize) -> Vec<[usize; 2]> {
        let mut res = Vec::new();
        for i in 0..self.size as usize {
            res.push([i, (i + distance) % self.size as usize]);
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

        let mut dist = rand::thread_rng().gen_range(1..self.size);
        let mut previous_dists = vec![dist];

        // Precompute pairings outside the parallel loop to avoid mutating `self`
        let mut pairings = Vec::with_capacity(games); // Preallocate space for efficiency
        for _ in 0..games {
            pairings.push(self.circular_pairing(dist as usize));
            let dist_old = rand::thread_rng().gen_range(1..self.size);
            dist = dist_old;
            while !previous_dists.iter().all(|&factor| dist % self.size % factor != 0) && dist < self.size + dist_old {
                dist += 1;
            }
            dist = dist % self.size;
            previous_dists.push(dist);
        }

        // Mutex for safe concurrent updates of agent fitness
        let fitness_updates = Arc::new(Mutex::new(vec![0.0; self.agents.len()]));

        let mut count = AtomicUsize::new(0);
        let mut moves_sum = AtomicU32::new(0);

        // Use `rayon`'s thread pool to run games in parallel
        pairings.into_par_iter().for_each(|pairing| {
            bar.inc(1);
            for j in pairing {
                let agents = Arc::clone(&agents); // Only clone if needed
                let game = Arc::clone(&game_arc); // Only clone if needed
                let fitness_updates = Arc::clone(&fitness_updates); // Clone fitness mutex

                // Run the game and get the results
                let agents_slice = vec![&agents[j[0]], &agents[j[1]]];
                let game_res = run_game(&game, agents_slice, false, MOVE_FITNESS);

                // Update fitness in a critical section
                let mut fitness_lock = fitness_updates.lock().unwrap();
                count.fetch_add(2, Ordering::SeqCst);
                moves_sum.fetch_add(game_res.2.iter().sum::<u32>(), Ordering::SeqCst);
                if self.comp_avg_moves >= 0. && MOVE_FITNESS {
                    fitness_lock[j[0]] += game_res.0[0] as f64 + 1. / (game_res.2[0].max(1) as f64).powf(2.) * self.comp_avg_moves;
                    fitness_lock[j[1]] += game_res.0[1] as f64 + 1. / (game_res.2[1].max(1) as f64).powf(2.) * self.comp_avg_moves;
                } else {
                    fitness_lock[j[0]] += game_res.0[0] as f64;
                    fitness_lock[j[1]] += game_res.0[1] as f64;
                }
            }
        });

        // After all threads finish, update the agents' fitness from the results
        let fitness_lock = fitness_updates.lock().unwrap();
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.fitness = fitness_lock[i];
        }
        self.comp_avg_moves = moves_sum.load(Ordering::SeqCst) as f64 / count.load(Ordering::SeqCst) as f64;
    }

    //compete_best_agents() but multithreaded
    pub fn compete_best_agents_mt(&mut self, game: &T, agent: &Agent) -> (Vec<u32>, Vec<(usize, usize, GameResLog)>) {
        let game_arc = Arc::new(game.clone());
        let best_agents_arc = Arc::new(self.best_agents.clone()); // Clone `self.best_agents` into the `Arc`

        if best_agents_arc.len() == 0 {
            return (vec![], vec![]);
        }

        let step = best_agents_arc.len() as f64 / BEST_AGENT_TOURNAMENT_MAX as f64;
        let mut count = best_agents_arc.len() as f64 % step;

        // Prepare pairings upfront for easier parallel processing
        let mut pairings = Vec::new();

        println!("Pairings:");
        print!("[");
        let mut k = 0;
        while count < best_agents_arc.len() as f64 - 1. + step {
            let i = (count as usize).min(best_agents_arc.len()- 1);
            print!("{}, ", i);
            let best_agents = Arc::clone(&best_agents_arc); // Clone the `Arc` to use inside the closure

            for j in 0..2 {
                // Dereference `best_agents` and clone the required agents from the underlying Vec
                let mut agents = vec![agent.clone(), best_agents[i].clone()];
                if j == 1 {
                    agents.reverse();
                }
                pairings.push((k, i, j, agents));
                k += 1;
            }
            count += step;
        }
        println!("]");
        

        let print_games_mutex = Arc::new(Mutex::new(Vec::new()));
        let mut comp_moves_mutex = Arc::new(Mutex::new(vec![0; pairings.len()]));
        let mut comp_performances_mutex = Arc::new(Mutex::new(vec![0; pairings.len()]));
        let comp_performance_sum_mutex = Arc::new(Mutex::new(0));
        let comp_moves_sum_mutex = Arc::new(Mutex::new(0));


        let print_agent = rand::thread_rng().gen_range(0..pairings.len());
        let run_game = self.run_game;

        self.best_agents_comp_res = vec![0; pairings.len()];

        // Mutex to protect the shared results
        let results_mutex = Arc::new(Mutex::new(vec![0; pairings.len() / 2]));


        // Use `rayon` for parallel execution
        pairings.clone().into_par_iter().for_each(|(k, i, j, agents)| {
            let game = Arc::clone(&game_arc);
            let results_mutex = Arc::clone(&results_mutex);
            let print_games_mutex = Arc::clone(&print_games_mutex);
            let comp_moves_mutex = Arc::clone(&comp_moves_mutex);
            let comp_performances_mutex = Arc::clone(&comp_performances_mutex);
            let comp_performance_sum_mutex = Arc::clone(&comp_performance_sum_mutex);
            let comp_moves_sum_mutex = Arc::clone(&comp_moves_sum_mutex);

            // Decide whether to print or not
            let game_res_log = if k == print_agent || i == best_agents_arc.len() - 1 {
                let agents_ref = vec![&agents[0], &agents[1]];
                let mut str = String::new();
                if j == 0 {
                    writeln!(&mut str, "best_agent vs agent: {}", i).unwrap();
                } else {
                    writeln!(&mut str, "agent: {} vs best_agent", i).unwrap();
                }
                let res = run_game(&game, agents_ref, true, true);
                for h in 0..res.1.len() {
                    let (state, agent_move) = res.1[h].clone();
                    writeln!(&mut str, "Turn {}: state: {:?}, agent_move: {:?}", h, state, agent_move).unwrap();
                }
                writeln!(&mut str, "game_res: {:?}", res.0).unwrap();
                println!("{}", str);
                res
            } else {
                let agents_ref = vec![&agents[0], &agents[1]];
                run_game(&game, agents_ref, false, true)
            };

            let game_res = game_res_log.clone().0;

            // Update the results, moves and performances safely
            let mut results_lock = results_mutex.lock().unwrap();
            results_lock[k / 2] += game_res[j];
            if k == print_agent || i == best_agents_arc.len() - 1 {
                let mut print_games_lock = print_games_mutex.lock().unwrap();
                if j == 0 {
                    print_games_lock.push((self.cycle as usize, i, game_res_log.clone()));
                } else {
                    print_games_lock.push((i, self.cycle as usize, game_res_log.clone()));
                }
            }
            let mut comp_moves_lock = comp_moves_mutex.lock().unwrap();
            comp_moves_lock[k] = game_res_log.2.iter().sum::<u32>();
            let mut comp_performances_lock = comp_performances_mutex.lock().unwrap();
            comp_performances_lock[k] = game_res_log.3[j] / game_res_log.2[j].max(1);
            let mut comp_performance_sum_lock = comp_performance_sum_mutex.lock().unwrap();
            *comp_performance_sum_lock += game_res_log.3[j];
            let mut comp_moves_sum_lock = comp_moves_sum_mutex.lock().unwrap();
            *comp_moves_sum_lock += game_res_log.2[j];
        });

        // Retrieve the final results from the mutex
        let final_results = results_mutex.lock().unwrap().clone();
        self.best_agents_comp_res = final_results.clone();
        self.best_agents_won_games = self.best_agents_comp_res.iter().sum();
        self.best_agent_played_games = pairings.len() as u32;
        let print_games = print_games_mutex.lock().unwrap().clone();
        self.best_agents_comp_moves = comp_moves_mutex.lock().unwrap().clone();
        self.best_agent_performances = comp_performances_mutex.lock().unwrap().clone();

        // Calculate the averages
        self.best_agents_comp_avg_moves = self.best_agents_comp_moves.iter().sum::<u32>() as f64 / self.best_agents_comp_moves.len() as f64;
        //self.best_agent_avg_performances = self.best_agent_performances.iter().sum::<u32>() as f64 / self.best_agent_performances.len() as f64;
        self.best_agent_avg_performances = comp_performance_sum_mutex.lock().unwrap().clone() as f64 / comp_moves_sum_mutex.lock().unwrap().clone() as f64;

        (final_results, print_games)
    }


    pub fn evolve(&mut self) {
        let mut new_agents = Vec::new();
        //add old best agents to new population
        let mut best_agents = self.best_agents.clone();
        for _ in 0..self.size / 100 * BEST_AGENT_SHARE {
            if !best_agents.is_empty() {
                let i = rand::random::<u64>() as usize % best_agents.len();
                new_agents.push(best_agents[i].clone().mutate((rand::random::<f64>() * (self.mutation_rate_range.1 - self.mutation_rate_range.0) as f64 + self.mutation_rate_range.0 as f64) as usize));
            }
        }

        //add random old agent to new population
        for _ in 0..self.size / 100 * RANDOM_OLD_AGENT_SHARE {
            new_agents.push(self.agents[(rand::random::<u32>() % self.size) as usize].clone().mutate((rand::random::<f64>() * (self.mutation_rate_range.1 - self.mutation_rate_range.0) as f64 + self.mutation_rate_range.0 as f64) as usize));
        }

        //add random agents to new population
        for _ in 0..self.size / 100 * RANDOM_AGENT_SHARE {
            new_agents.push(Agent::new(self.inputs, self.outputs));
        }

        //crossover and mutate new agents
        let mut agents_fitness = Vec::new();
        for agent in &self.agents {
            agents_fitness.push(agent.fitness.powf(FITNESS_EXP));
        }
        let total_fitness: f64 = agents_fitness.iter().sum();
        while new_agents.len() < self.size as usize {
            let mut fitness_index = rand::random::<f64>() * total_fitness;
            let mut index = 0;
            while fitness_index > agents_fitness[index] {
                fitness_index -= agents_fitness[index];
                index += 1;
            }
            let mut agent = self.agents[index].clone();
            agent.mutate((rand::random::<f64>() * (self.mutation_rate_range.1 - self.mutation_rate_range.0) as f64 + self.mutation_rate_range.0 as f64) as usize);
            new_agents.push(agent);
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

        match wtr.write_record(["generation", "time", "avg_layers", "avg_hidden_layer_size", "avg_moves", /*"best_agent_layer_sizes",*/ "best_agent_fitness", "best_agent_wins_percentage", "best_agent_avg_performance"]) {
            Ok(_) => Ok(()),
            Err(err) => Err(Box::new(err)),
        }
    }

    pub fn save_stats_csv(&self, filename: &str) -> csv::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .unwrap();
        let mut wtr = csv::Writer::from_writer(file);
        let agents = &self.agents;
        let mut avg_layers = 0.;
        let mut avg_hidden_layers = 0.;
        let mut avg_hidden_layer_size = 0.;
        for agent in agents.iter() {
            avg_layers += agent.nn.layer_sizes.len() as f64;
            for i in 1..agent.nn.layer_sizes.len() - 1 {
                avg_hidden_layer_size += agent.nn.layer_sizes[i] as f64;
                avg_hidden_layers += 1.;
            }
        }
        avg_layers /= self.agents.len() as f64;
        avg_hidden_layer_size /= f64::max(avg_hidden_layers, 1.);

        let mut best_agent_layer_sizes = "".to_owned();
        for i in self.agents[0].nn.layer_sizes.clone() {
            best_agent_layer_sizes.push_str(&format!("{},", i).to_string());
        }

        wtr.write_record(&[
            self.cycle.to_string(),
            self.total_elapsed.to_string(),
            avg_layers.to_string(),
            avg_hidden_layer_size.to_string(),
            self.best_agents_comp_avg_moves.to_string(),
            //best_agent_layer_sizes,
            self.agents[0].fitness.to_string(),
            (self.best_agents_won_games as f64 / self.best_agent_played_games as f64 * 100.).to_string(),
            (self.best_agent_avg_performances * 100.).to_string()]
        )
    }

    pub fn create_best_agent_games_txt(&self, filename: &str) {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(filename)
            .unwrap();
    }
    pub fn save_best_agent_games_txt(&self, filename: &str, games: Vec<(usize, usize, GameResLog)>) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(filename)
            .unwrap();

        writeln!(file, "Generation: {}", self.cycle).unwrap();
        writeln!(file, "Best agent layer sizes: {:?}", self.agents[0].nn.layer_sizes).unwrap();
        for i in games {
            writeln!(file, "Agent {} vs Agent {}:", i.0, i.1).unwrap();
            for j_index in 0..i.2.1.len() {
                let j = &i.2.1[j_index];
                writeln!(file, "Turn: {}: state: {:?}, agent_move: {:?}", j_index, j.0, j.1).unwrap();
            }
            writeln!(file, "Game result: {:?}", i.2.0).unwrap();
        }
        writeln!(file).unwrap();
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

    pub fn save_params_csv(&self, filename: &str, func_str: &str, comp_games: usize, initial_state: Vec<u32>) {
        let mut wtr = csv::Writer::from_path(filename).unwrap();

        wtr.write_record([
            "game func",
            "initial state",
            "population size",
            "comp games",
            "mutation min",
            "mutation max",
            "fitness exponent",
            "best agent share",
            "random agent share",
            "random old agent share",
            "best agent tournament games",
            "add_connection_rand",
            "add_node_rand",
            "change_weight_rand",
            "change_bias_rand",
            "shift_weight_rand",
            "shift_bias_rand"
        ]).expect("CSV Writer Error!");

        let mut initial_state_str = String::new();
        write!(initial_state_str, "{:?}", initial_state).expect("write didnt work");

        wtr.write_record([
            func_str.to_string(),
            initial_state_str,
            self.size.to_string(),
            comp_games.to_string(),
            self.mutation_rate_range.0.to_string(),
            self.mutation_rate_range.1.to_string(),
            FITNESS_EXP.to_string(),
            BEST_AGENT_SHARE.to_string(),
            RANDOM_AGENT_SHARE.to_string(),
            RANDOM_OLD_AGENT_SHARE.to_string(),
            BEST_AGENT_TOURNAMENT_MAX.to_string(),
            MUTATION_TYPES[0].weight.to_string(),
            MUTATION_TYPES[1].weight.to_string(),
            MUTATION_TYPES[2].weight.to_string(),
            MUTATION_TYPES[3].weight.to_string(),
            MUTATION_TYPES[4].weight.to_string(),
            MUTATION_TYPES[5].weight.to_string(),
        ]).expect("CSV Writer Error!")
    }
}
