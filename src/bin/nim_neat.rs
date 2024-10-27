use rust_neat::neat::population::Population;
use rust_neat::nim::nim::Nim;
use std::thread;

const USE_BIN: bool = false;

fn main() {
    let nim_func = Nim::run_nim_strict_random;
    //let best_agent_tournament_csv = "best_agent_tournament_single.csv";
    let stats_csv = "stats_single.csv";
    let best_agent_games_txt = "best_agent_games_single.txt";
    let nim_config = Nim::new(vec![10]);
    println!("input size: {}, output size: {}", nim_config.input_size, nim_config.output_size);
    
    //start timer
    let now = std::time::Instant::now();
    let mut last = now.clone();
    let mut population = if USE_BIN {
        Population::load_population("population.bin").unwrap()
    } else {
        Population::new(200, nim_config.input_size, nim_config.output_size, nim_func, (1usize, 5usize))
        //Population::new(20, nim_config.input_size, nim_config.output_size, Nim::run_nim, (1usize, 5usize));
    }; 
    if USE_BIN {
        population.run_game = nim_func;
    } else {
        //population.create_best_agent_tournament_csv(best_agent_tournament_csv);
        population.create_stats_csv(stats_csv);
        population.create_best_agent_games_txt(best_agent_games_txt);
    }
    let mut saver= thread::spawn(|| {Ok(())});
    for _ in 0..10000 {
        //let mut population = Population::load_population("population.bin").unwrap();
        //population.run_game = Nim::run_nim_strict;
        
        println!("generation: {}", population.cycle);
        population.competition(&nim_config, 75);
        population.rank_agents();

        // Limit the scope of the immutable borrow
        let best_agent = {
            population.agents[0].clone()
        };


        //saver.join().unwrap();
        let res = population.compete_best_agents_mt(&nim_config, &best_agent);
        if saver.is_finished() {
            saver.join().unwrap();
            saver = population.save_population("population.bin");
            println!("Saving population {}!", population.cycle);
        }
        println!("best fitness: {}", best_agent.fitness);
        //println!("best agent: {:?}", best_agent);
        println!("layer_sizes: {:?}", best_agent.nn.layer_sizes);
        println!("edge_count: {}", best_agent.nn.edge_count);
        //print the competition results in green if 2, in white if 1 and in red if 0
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
        println!("Time elapsed: {:?}", now.elapsed());
        println!("Time since last: {:?}", last.elapsed());
        last = std::time::Instant::now();
        //population.save_best_agent_tournament_csv(best_agent_tournament_csv);
        population.save_stats_csv(stats_csv);
        population.save_best_agent_games_txt(best_agent_games_txt, res.1);
        population.best_agents.push(best_agent.clone());
        population.evolve();  // Now you can mutate population
    }

}
