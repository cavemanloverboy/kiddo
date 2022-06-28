extern crate kiddo;

use kiddo::distance::squared_euclidean;
use kiddo::KdTree;
use std::sync::atomic::{AtomicUsize, Ordering};

static POINT_A: ([f64; 2], usize) = ([0f64, 0f64], 0);
static POINT_B: ([f64; 2], usize) = ([1f64, 1f64], 1);
static POINT_C: ([f64; 2], usize) = ([2f64, 2f64], 2);
static POINT_D: ([f64; 2], usize) = ([3f64, 3f64], 3);

#[test]
fn it_works() {
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_per_node_capacity(capacity_per_node).unwrap();

    let count = AtomicUsize::new(0);
    let new_dist = |a: &[f64; 2], b: &[f64; 2]| {
        count.fetch_add(1, Ordering::SeqCst);
        squared_euclidean(a, b)
    };

    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_D.0, POINT_D.1).unwrap();

    kdtree.nearest(&POINT_A.0, 0, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 0);

    kdtree.nearest(&POINT_A.0, 1, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 2);

    kdtree.nearest(&POINT_A.0, 2, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 4);

    kdtree.nearest(&POINT_A.0, 3, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_A.0, 4, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_A.0, 5, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_B.0, 4, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 6);

    kdtree.within(&POINT_A.0, 0.0, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 2);

    kdtree.within(&POINT_B.0, 1.0, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 3);

    kdtree.within(&POINT_B.0, 2.0, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 6);

    let mut iter = kdtree.iter_nearest(&POINT_A.0, &new_dist).unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 0);

    iter.next().unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(count.swap(0, Ordering::SeqCst), 0);
}
