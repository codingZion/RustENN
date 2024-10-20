pub fn get_bases(nim_heaps: Vec<usize>) -> Vec<usize> {
    let mut bases = Vec::new();
    let mut base = 1;
    for i in nim_heaps.iter() {
        bases.push(base);
        base *= i + 1;
    }
    bases
}

pub fn get_index(nim_heap_state: Vec<usize>, bases: &Vec<usize>) -> usize {
    let mut index = 0;
    for i in nim_heap_state.iter().enumerate() {
        index += i.1 * bases[i.0];
    }
    index
}

pub fn get_state(index: usize, nim_heaps: Vec<usize>) -> Vec<usize> {
    let mut state = Vec::new();
    let mut index = index;
    for i in nim_heaps.iter() {
        let cur = index % (i + 1);
        state.push(cur);
        index -= cur;
        index /= i + 1;
    }
    state
}

pub fn dp_alg(nim_heap_state: Vec<usize>, nim_heaps: Vec<usize>, dp: &mut Vec<(i32, isize, bool)>, bases: &Vec<usize>) -> (i32, isize, bool) {
    let cur_index = get_index(nim_heap_state.clone(), bases, );
    if dp[cur_index].0 != -1 {
        return dp[cur_index];
    }
    let mut res = (-1, -1, false);
    for i in nim_heap_state.iter().enumerate() {
        if i.1.clone() != 0usize {
            for j in 1..i.1 + 1 {
                let mut new_state = nim_heap_state.clone();
                new_state[i.0] -= j;
                let mut new_res = dp_alg(new_state.clone(), nim_heaps.clone(), dp, bases);
                new_res.0 += 1;
                new_res.1 = get_index(new_state.clone(), bases) as isize;
                new_res.2 = !new_res.2;
                if res.0 == -1 {
                    res = new_res;
                } else if new_res.2 {
                    if !res.2 || (res.2 && new_res.0 < res.0) {
                        res = new_res;
                    }
                } else if !res.2 && res.0 < new_res.0 {
                    res = new_res;
                }
            }
        }
    }
    dp[cur_index] = res;
    res
}

pub fn get_move(nim_heap_state: Vec<usize>, nim_heaps: Vec<usize>, dp: &Vec<(i32, isize, bool)>, bases: &Vec<usize>) -> Vec<usize> {
    let cur_index = get_index(nim_heap_state.clone(), bases, );
    let res = dp[cur_index].1;
    get_state(res as usize, nim_heaps.clone())
}

pub fn print_dp(dp: &Vec<(i32, isize, bool)>, nim_heaps: Vec<usize>) {
    for i in 0..dp.len() {
        println!("{} - {:?}: {:?}", i, get_state(i, nim_heaps.clone()), dp[i]);
    }
}