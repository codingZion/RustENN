use rust_neat::neat::population::Population;
use rust_neat::nim::nim2::Nim;
use std::{fs, thread};
use std::time::SystemTime;
//mute println
/*
macro_rules! println {
    () => {};
    ($($arg:tt)*) => {};
}
*/

const USE_BIN: bool = false;

fn main() {
    let nim_func = Nim::run_nim_strict_random;
    let func_str = "run_nim2_strict_random";
    let comp_games = 50;
    let initial_state = vec![4;2];
    let dir_name = "data_out/";
    fs::create_dir_all(dir_name).expect("Couldn't create Directory");
    //let best_agent_tournament_csv = "best_agent_tournament_single.csv";
    let stats_csv = "stats.csv";
    let best_agent_games_txt = "best_agent_games.txt";
    let params_csv = "params.csv";
    let population_file = "population.bin";
    let tmp_prefix = "z_";
    let nim_config = Nim::new(initial_state.clone());
    println!("input size: {}, output size: {}", nim_config.input_size, nim_config.output_size);
    
    //start timer
    let mut last = SystemTime::now();
    let mut population = if USE_BIN {
        Population::load_population(dir_name.to_owned() + "population.bin").unwrap()
    } else {
        Population::new(2500, nim_config.input_size, nim_config.output_size, nim_func, (0usize, 1usize))
        //Population::new(20, nim_config.input_size, nim_config.output_size, Nim::run_nim, (1usize, 5usize));
    }; 
    if USE_BIN {
        population.run_game = nim_func;
    } else {
        //population.create_best_agent_tournament_csv(best_agent_tournament_csv);
        population.create_stats_csv(dir_name.to_owned() + stats_csv).expect("CSV couldn't be created!");
        population.create_best_agent_games_txt(dir_name.to_owned() + best_agent_games_txt);
    }
    let mut saver= thread::spawn(|| {Ok(())});

    population.save_params_csv(dir_name.to_owned() + params_csv, func_str, comp_games, initial_state.clone());

    for _ in 0..10000 {
        //let mut population = Population::load_population("population.bin").unwrap();
        //population.run_game = Nim::run_nim_strict;
        
        println!("generation: {}", population.cycle);
        population.competition(&nim_config, comp_games);
        population.rank_agents();

        // Limit the scope of the immutable borrow
        let best_agent = {
            population.agents[0].clone()
        };


        //saver.join().unwrap();
        let res = population.compete_best_agents(&nim_config, &best_agent);
        if saver.is_finished() {
            saver.join().unwrap_or(Ok(())).expect("Saving failed!");
            saver = population.save_population(population_file, tmp_prefix, dir_name.parse().unwrap());
            println!("Saving population {}!", population.cycle);
        }
        println!("best fitness: {}", best_agent.fitness);
        //println!("best agent: {:?}", best_agent);
        println!("layer_sizes: {:?}", best_agent.nn.layer_sizes);
        println!("edge_count: {}", best_agent.nn.edge_count);
        print!("Competition results: ");
        for i in res.0.iter() {
            if *i == 2 {
                print!("\x1b[32m{}\x1b[0m", i);
            } else if *i == 1 {
                print!("{}", i);
            } else {
                print!("\x1b[31m{}\x1b[0m", i);
            }
        }
        println!();
        //println!("Competition results: {:?}", res);
        println!("Wins: {:?} -> {:?}%", res.0.iter().sum::<u32>(), res.0.iter().sum::<u32>() as f64 / res.0.len() as f64 * 50.);
        println!("Time elapsed: {:?}", population.total_elapsed);
        println!("Time since last: {:?}", last.elapsed().unwrap().as_secs_f32());
        population.total_elapsed += last.elapsed().unwrap().as_secs_f64();
        last = SystemTime::now();
        //population.save_best_agent_tournament_csv(best_agent_tournament_csv);
        population.save_stats_csv(dir_name.to_owned() + stats_csv);
        population.save_best_agent_games_txt(dir_name.to_owned() + best_agent_games_txt, res.1);
        population.best_agents.push(best_agent.clone());
        population.evolve();  // Now you can mutate population
    }
    saver.join().unwrap_or(Ok(())).expect("Saving failed!");
    population.save_population(population_file, tmp_prefix, dir_name.parse().unwrap());
    let tmp_file = dir_name.to_owned() + &*tmp_prefix.to_owned() + &*population_file.to_owned() + ".tmp";
    println!("{}", tmp_file);
    fs::remove_file(tmp_file).expect("File removal failed");
}
