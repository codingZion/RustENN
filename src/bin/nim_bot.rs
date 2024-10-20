use rust_neat::nim::dp;

fn main() {
    let nim_heaps = vec![3, 4, 5, 6, 7];
    //let mut dp = ndarray::ArrayD::<Vec<i32>>::default(ndarray::IxDyn(&[3, 4, 5]));
    let states_count = nim_heaps.iter().map(|&x| x + 1).product();
    println!("{}", states_count);
    let mut dp = vec![(-1, -1, false); states_count];
    dp[0] = (0, 0, true);
    dp::dp_alg(nim_heaps.clone(), nim_heaps.clone(), &mut dp, &dp::get_bases(nim_heaps.clone()));
    println!("{:?}", dp[dp::get_index(nim_heaps.clone(), &nim_heaps.clone())]);
    println!("{:?}", dp::get_move(nim_heaps.clone(), nim_heaps.clone(), &dp, &dp::get_bases(nim_heaps.clone())));
    dp::print_dp(&dp, nim_heaps.clone());
}
