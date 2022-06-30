use kiddo::{KdTree, ErrorKind, distance::squared_euclidean};
use rand::distributions::{Distribution, Uniform};
use rand::{thread_rng, rngs::ThreadRng};
use rayon::prelude::*;
use std::sync::Arc;

type Float = f64;
const K: usize = 3;
const RUNS: u128 = 100;
const NRAND: usize = 1_000_000;
const NQUERY: usize = 100_000;

fn main() -> Result<(), ErrorKind> {

    let threads = 1; 
    // let threads = std::thread::available_parallelism().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    println!("using {threads} threads");
    for p_flag in 0..2 {

        let mut tree_building = 0;
        let mut millis_queriying = 0;
        for _ in 0..RUNS {

            // Initialize rng
            let mut rng = thread_rng();
            let distr = Uniform::<Float>::new(0.0, 1.0);

            // Initialize periodic kdtree
            let periodic: [Float; 3] = [1.0, 1.0, 1.0];
            // let mut periodic_tree = KdTree::new_periodic(periodic);
            let mut periodic_tree = if p_flag == 0 { KdTree::with_per_node_capacity(32)?} else {KdTree::periodic_with_per_node_capacity(32, periodic)?} ;
            
            // Populate tree
            let local_timer = std::time::Instant::now();
            for i in 0..10_usize.pow(5) {
                periodic_tree.add(&sample_cube::<K>(distr, &mut rng), i).expect("failed to add point to tree");
            }
            tree_building += local_timer.elapsed().as_millis();

            // Query tree
            // let periodic_tree = Arc::new(periodic_tree);
            let rands: Vec<[Float; K]> = (0..NQUERY)
                .map(|_| sample_cube::<K>(distr, &mut rng))
                .collect::<Vec<_>>();

            let local_timer = std::time::Instant::now();
            let mut knns: [(Float, &usize); NRAND] = [(0.0, &0); NRAND];
            for (i, query) in rands.iter().enumerate() {
                knns[i] = periodic_tree.nearest_one(&query, &squared_euclidean).unwrap();
            }

            millis_queriying += local_timer.elapsed().as_millis();
    }

    println!("finished with {} millis average querying, {} building", millis_queriying/RUNS, tree_building/RUNS);
    }
    Ok(())
}


fn sample_cube<const K: usize>(distr: Uniform<Float>, rng: &mut ThreadRng) -> [Float; K] {

    // Initialize point
    let mut point = [0.0 as Float; K];

    // Populate components
    for idx in 0..K {
        point[idx] = distr.sample(rng);
    }

    // Return point
    point
}


fn get_local_set<'a, T: Clone>(set: &'a [T], threads: usize, id: usize) -> Vec<T> {

    let sets = set.chunks(div_ceil(set.len(), threads))
        .map(|x| x.to_vec())
        .collect::<Vec<Vec<T>>>();
    println!("sets len is {}", sets.len());
    sets
        .get(id)
        .unwrap()
        .clone()
}


fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}