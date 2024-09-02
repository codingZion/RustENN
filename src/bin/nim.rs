use rust_neat::neat::population::Population;
use rust_neat::neat::agent::Agent;
use rust_neat::nim::nim::Nim;
use std::sync::Arc;
use std::sync::Mutex;

fn main() {
    let mut nim_config = Nim::new(vec![3, 4, 5, 6, 7]);
    println!("input size: {}, output size: {}", nim_config.input_size, nim_config.output_size);
    
    //start timer
    let now = std::time::Instant::now();
    let mut last = now.clone();
    let mut population = Population::new(20000, nim_config.input_size, nim_config.output_size, Nim::run_nim_strict, (1usize, 5usize));
    //let mut population = Population::new(20, nim_config.input_size, nim_config.output_size, Nim::run_nim, (1usize, 5usize));
    population.create_best_agent_tournament_csv("best_agent_tournament.csv");
    population.create_stats_csv("stats.csv");

    for i in 0..10000 {
        println!("generation: {}", i);
        population.competition(&nim_config, 75);
        population.rank_agents();

        // Limit the scope of the immutable borrow
        let best_agent = {
            population.agents[0].clone()
        };


        let res = population.compete_best_agents_mt(&nim_config, &best_agent);
        println!("best fitness: {}", best_agent.fitness);
        //println!("best agent: {:?}", best_agent);
        println!("layer_sizes: {:?}", best_agent.nn.layer_sizes);
        println!("edge_count: {}", best_agent.nn.edge_count);
        //print the competition results in green if 2, in white if 1 and in red if 0
        print!("Competition results: ");
        for i in res.iter() {
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
        println!("Wins: {:?} -> {:?}%", res.iter().sum::<u32>(), res.iter().sum::<u32>() as f64 / res.len() as f64 * 50.);
        println!("Time elapsed: {:?}", now.elapsed());
        println!("Time since last: {:?}", last.elapsed());
        last = std::time::Instant::now();
        population.save_best_agent_tournament_csv("best_agent_tournament.csv");
        population.save_stats_csv("stats.csv");
        population.best_agents.lock().unwrap().push(best_agent.clone());
        population.evolve();  // Now you can mutate population
    }

}
