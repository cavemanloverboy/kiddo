use std::collections::BinaryHeap;

use num_traits::{Float, One, Zero};

#[cfg(feature = "serialize")]
use crate::custom_serde::*;
use crate::heap_element::HeapElement;
use crate::util;
use num_traits::Signed;

trait Stack<T>
where
    T: Ord,
{
    fn stack_push(&mut self, _: T);
    fn stack_pop(&mut self) -> Option<T>;
}

impl<T> Stack<T> for Vec<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        Vec::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> Stack<T> for BinaryHeap<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        BinaryHeap::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        BinaryHeap::<T>::pop(self)
    }
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct KdTree<A, T: std::cmp::PartialEq, const K: usize> {
    size: usize,

    #[cfg_attr(feature = "serialize", serde(with = "arrays"))]
    min_bounds: [A; K],
    #[cfg_attr(feature = "serialize", serde(with = "arrays"))]
    max_bounds: [A; K],
    content: Node<A, T, K>,
    periodic: Option<[A; K]>,
}

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum Node<A, T: std::cmp::PartialEq, const K: usize> {
    Stem {
        left: Box<KdTree<A, T, K>>,
        right: Box<KdTree<A, T, K>>,
        split_value: A,
        split_dimension: u8,
    },
    Leaf {
        #[cfg_attr(feature = "serialize", serde(with = "vec_arrays"))]
        points: Vec<[A; K]>,
        bucket: Vec<T>,
        capacity: usize,
    },
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    PeriodicOutOfBounds,
    NonFiniteCoordinate,
    ZeroCapacity,
    Empty,
}

impl<A: Float + Zero + One + Signed, T: std::cmp::PartialEq, const K: usize> KdTree<A, T, K> {
    /// Creates a new KdTree with default capacity **per node** of 16.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn new() -> Self {
        KdTree::with_per_node_capacity(16).unwrap()
    }

    /// Creates a new KdTree with default capacity **per node** of 16, with periodic boundary conditions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new_periodic([5.0,5.0,5.0]);
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn new_periodic(periodic: [A; K]) -> Self {
        KdTree::periodic_with_per_node_capacity(16, periodic).unwrap()
    }

    /// Creates a new KdTree with a specific capacity **per node**. You may wish to
    /// experiment by tuning this value to best suit your workload via benchmarking:
    /// values between 10 and 40 often work best.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::with_per_node_capacity(30)?;
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn with_per_node_capacity(capacity: usize) -> Result<Self, ErrorKind> {
        if capacity == 0 {
            return Err(ErrorKind::ZeroCapacity);
        }

        Ok(KdTree {
            size: 0,
            min_bounds: [A::infinity(); K],
            max_bounds: [A::neg_infinity(); K],
            content: Node::Leaf {
                points: Vec::with_capacity(capacity),
                bucket: Vec::with_capacity(capacity),
                capacity,
            },
            periodic: None,
        })
    }

    /// Creates a new KdTree with a specific capacity **per node**, and wih perodic
    /// boundary conditions. You may wish to experiment by tuning this value to 
    /// best suit your workload via benchmarking: values between 10 and 40 often 
    /// work best.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = 
    ///     KdTree::periodic_with_per_node_capacity(30, [6.0, 6.0, 6.0])?;
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn periodic_with_per_node_capacity(capacity: usize, periodic: [A; K]) -> Result<Self, ErrorKind> {
        if capacity == 0 {
            return Err(ErrorKind::ZeroCapacity);
        }

        Ok(KdTree {
            size: 0,
            min_bounds: [A::infinity(); K],
            max_bounds: [A::neg_infinity(); K],
            content: Node::Leaf {
                points: Vec::with_capacity(capacity),
                bucket: Vec::with_capacity(capacity),
                capacity,
            },
            periodic: Some(periodic),
        })
    }

    /// Creates a new KdTree with a specific capacity **per node**.
    ///
    #[deprecated(since = "0.1.8", note = "with_capacity has a misleading name. Users should instead use with_per_node_capacity. with_capacity will be removed in a future release")]
    pub fn with_capacity(capacity: usize) -> Result<Self, ErrorKind> {
        Self::with_per_node_capacity(capacity)
    }

    /// Returns the current number of elements stored in the tree
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree.size(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns true if the node is a leaf node
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree_1: KdTree<f64, usize, 3> = KdTree::with_capacity(2)?;
    ///
    /// tree_1.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree_1.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree_1.is_leaf(), true);
    ///
    /// let mut tree_2: KdTree<f64, usize, 3> = KdTree::with_capacity(1)?;
    ///
    /// tree_2.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree_2.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree_2.is_leaf(), false);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn is_leaf(&self) -> bool {
        match &self.content {
            Node::Leaf { .. } => true,
            Node::Stem { .. } => false,
        }
    }

    /// Queries the tree to find the nearest `num` elements to `point`, using the specified
    /// distance metric function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest(&[1.0, 2.0, 5.1], 1, &squared_euclidean)?;
    ///
    /// assert_eq!(nearest.len(), 1);
    /// assert!((nearest[0].0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest[0].1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn nearest<F>(
        &self,
        point: &[A; K],
        num: usize,
        distance: &F,
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.check_point(point)?;

        let num = std::cmp::min(num, self.size);
        if num == 0 {
            return Ok(vec![]);
        }

        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty()
            && (evaluated.len() < num
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(
                point,
                num,
                A::infinity(),
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated
            .into_sorted_vec()
            .into_iter()
            .take(num)
            .map(Into::into)
            .collect())
    }

    /// Queries the tree to find the nearest `num` elements to `point`, using the specified
    /// distance metric function. Obeys periodic boundary conditions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// const PERIODIC: [f64; 3] = [10.0, 10.0, 10.0];
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest_periodic(&[1.0, 2.0, 5.1], 1, &squared_euclidean, &PERIODIC)?;
    ///
    /// assert_eq!(nearest.len(), 1);
    /// assert!((nearest[0].0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest[0].1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn nearest_periodic<F>(
        &self,
        point: &[A; K],
        num: usize,
        distance: &F,
        periodic: &[A; K],
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.check_point(point)?;

        let num = std::cmp::min(num, self.size);
        if num == 0 {
            return Ok(vec![]);
        }

        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty()
            && (evaluated.len() < num
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(
                point,
                num,
                A::infinity(),
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        // Find largest distance for canonical image
        let largest_distance: A = evaluated
            .iter()
            .fold(A::zero(), |acc, x| acc.max(x.distance));
        

        // Find closest dist2 to every side
        let mut closest_side_dist2: [A; K] = [A::zero(); K];
        for side in 0..K {

            // Do a single index here. This is equal to distance to lower side
            let query_component: A = point[side];

            // Get distance to upper half
            let upper = periodic[side] - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_negative()); 
            debug_assert!(!query_component.is_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with canonical image (which we will remove later)
        let mut images_to_check = Vec::with_capacity(2_usize.pow(K as u32)-1);
        for image in 1..2_usize.pow(K as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..K)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: A = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    Some(closest_side_dist2[side])
                } else { None })
                .fold(A::zero(), |acc, x| acc + x);

            if dist_to_side_edge_or_other < largest_distance {

                let mut image_to_check = point.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        let query_component: A = point[idx];
                        // Single index here as well
                        let periodic_component = periodic[idx];

                        if query_component < periodic_component / A::from(2_u8).unwrap() {
                            // Add if in lower half of box
                            image_to_check[idx] = query_component + periodic_component;
                        } else {
                            // Subtract if in upper half of box
                            image_to_check[idx] = query_component - periodic_component;
                        }
                        
                    }
                }

                images_to_check.push(image_to_check);
            }
        }

        // Then check all images
        for image in &images_to_check {
    
            let mut image_pending = BinaryHeap::new();
            let mut image_evaluated = BinaryHeap::<HeapElement<A, &T>>::new();
    
            image_pending.push(HeapElement {
                distance: A::zero(),
                element: self,
            });
    
            while !image_pending.is_empty()
                && (image_evaluated.len() < num
                    || (-image_pending.peek().unwrap().distance <= image_evaluated.peek().unwrap().distance))
            {
                self.nearest_step(
                    image,
                    num,
                    A::infinity(),
                    distance,
                    &mut image_pending,
                    &mut image_evaluated,
                );
            }

            evaluated.append(&mut image_evaluated);
        }

        Ok(evaluated
            .into_sorted_vec()
            .into_iter()
            .take(num)
            .map(Into::into)
            .collect())
    }

    /// Queries the tree to find the nearest element to `point`, using the specified
    /// distance metric function. Faster than querying for nearest(point, 1, ...) due
    /// to not needing to allocate a Vec for the result
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest_one(&[1.0, 2.0, 5.1], &squared_euclidean)?;
    ///
    /// assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest.1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    // TODO: pending only ever gets to about 7 items max. try doing this
    //       recursively to avoid the alloc/dealloc of the vec
    pub fn nearest_one<F>(&self, point: &[A; K], distance: &F) -> Result<(A, &T), ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Err(ErrorKind::Empty);
        }
        self.check_point(point)?;

        let mut pending = Vec::with_capacity(16);

        let mut best_dist: A = A::infinity();
        let mut best_elem: Option<&T> = None;

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() && (best_elem.is_none() || (pending[0].distance < best_dist)) {
            self.nearest_one_step(
                point,
                distance,
                &mut pending,
                &mut best_dist,
                &mut best_elem,
            );
        }

        Ok((best_dist, best_elem.unwrap()))
    }

        /// Queries the tree to find the nearest element to `point`, using the specified
    /// distance metric function. Faster than querying for nearest(point, 1, ...) due
    /// to not needing to allocate a Vec for the result
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// const PERIODIC: [f64; 3] = [10.0, 10.0, 10.0];
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let nearest = tree.nearest_one_periodic(&[1.0, 2.0, 5.1], &squared_euclidean, &PERIODIC)?;
    ///
    /// assert!((nearest.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest.1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    // TODO: pending only ever gets to about 7 items max. try doing this
    //       recursively to avoid the alloc/dealloc of the vec
    pub fn nearest_one_periodic<F>(&self, point: &[A; K], distance: &F, periodic: &[A; K]) -> Result<(A, &T), ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Err(ErrorKind::Empty);
        }
        self.check_point(point)?;

        let mut pending = Vec::with_capacity(16);

        let mut best_dist: A = A::infinity();
        let mut best_elem: Option<&T> = None;

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() && (best_elem.is_none() || (pending[0].distance < best_dist)) {
            self.nearest_one_step(
                point,
                distance,
                &mut pending,
                &mut best_dist,
                &mut best_elem,
            );
        }

        // Get canonical_image result as in non-PBC
        let canonical_image_result = (best_dist, best_elem.unwrap());

        // Find closest dist2 to every side
        let mut closest_side_dist2: [A; K] = [A::zero(); K];
        for side in 0..K {

            // Do a single index here. This is equal to distance to lower side
            let query_component: A = point[side];

            // Get distance to upper half
            let upper = periodic[side] - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_negative()); 
            debug_assert!(!query_component.is_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with canonical image (which we will remove later)
        let mut images_to_check = Vec::with_capacity(2_usize.pow(K as u32)-1);
        for image in 1..2_usize.pow(K as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..K)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: A = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    Some(closest_side_dist2[side])
                } else { None })
                .fold(A::zero(), |acc, x| acc + x);

            if dist_to_side_edge_or_other < canonical_image_result.0 {

                let mut image_to_check = point.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        let query_component: A = point[idx];
                        // Single index here as well
                        let periodic_component = periodic[idx];

                        if query_component < periodic_component / A::from(2_u8).unwrap() {
                            // Add if in lower half of box
                            image_to_check[idx] = query_component + periodic_component;
                        } else {
                            // Subtract if in upper half of box
                            image_to_check[idx] = query_component - periodic_component;
                        }
                        
                    }
                }

                images_to_check.push(image_to_check);
            }
        }

        // Then check all images
        for image in &images_to_check {

            let mut image_pending = Vec::with_capacity(16);

            let mut image_best_dist: A = A::infinity();
            let mut image_best_elem: Option<&T> = None;

            image_pending.push(HeapElement {
                distance: A::zero(),
                element: self,
            });

            while !image_pending.is_empty() && (image_best_elem.is_none() || (image_pending[0].distance < image_best_dist)) {
                self.nearest_one_step(
                    image,
                    distance,
                    &mut image_pending,
                    &mut image_best_dist,
                    &mut image_best_elem,
                );
            }

            if image_best_dist < best_dist {
                best_dist = image_best_dist;
                best_elem = image_best_elem;
            }
        }

        Ok((best_dist, best_elem.unwrap()))
    }

    fn within_impl<F>(
        &self,
        point: &[A; K],
        radius: A,
        distance: &F,
    ) -> Result<BinaryHeap<HeapElement<A, &T>>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.check_point(point)?;

        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() && (-pending.peek().unwrap().distance <= radius) {
            self.nearest_step(
                point,
                self.size,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated)
    }

    /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned sorted nearest-first
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within<F>(
        &self,
        point: &[A; K],
        radius: A,
        distance: &F,
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.within_impl(point, radius, distance).map(|evaluated| {
            evaluated
                .into_sorted_vec()
                .into_iter()
                .map(Into::into)
                .collect()
        })
    }

    /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned sorted nearest-first. Obeys periodic
    /// boundary conditions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// const PERIODIC: [f64; 3] = [10.0, 10.0, 10.0];
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within_periodic(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean, &PERIODIC)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within_periodic<F>(
        &self,
        point: &[A; K],
        radius: A,
        distance: &F,
        periodic: &[A; K],
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        // do as in within() but hold off on sorting
        let mut canonical_result: Vec<(A, &T)> = self.within_impl(point, radius, distance).map(|evaluated| {
            evaluated
                .into_vec()
                .into_iter()
                .map(Into::into)
                .collect()
        })?;


        // Find closest dist2 to every side
        let mut closest_side_dist2: [A; K] = [A::zero(); K];
        for side in 0..K {

            // Do a single index here. This is equal to distance to lower side
            let query_component: A = point[side];

            // Get distance to upper half
            let upper = periodic[side] - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_negative()); 
            debug_assert!(!query_component.is_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with canonical image (which we will remove later)
        let mut images_to_check = Vec::with_capacity(2_usize.pow(K as u32)-1);
        for image in 1..2_usize.pow(K as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..K)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: A = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    Some(closest_side_dist2[side])
                } else { None })
                .fold(A::zero(), |acc, x| acc + x);

            if dist_to_side_edge_or_other < radius {

                let mut image_to_check = point.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        let query_component: A = point[idx];
                        // Single index here as well
                        let periodic_component = periodic[idx];

                        if query_component < periodic_component / A::from(2_u8).unwrap() {
                            // Add if in lower half of box
                            image_to_check[idx] = query_component + periodic_component;
                        } else {
                            // Subtract if in upper half of box
                            image_to_check[idx] = query_component - periodic_component;
                        }
                        
                    }
                }

                images_to_check.push(image_to_check);
            }
        }

        // Then check all images
        for image in &images_to_check {

            let mut image_pending = Vec::with_capacity(16);

            image_pending.push(HeapElement {
                distance: A::zero(),
                element: self,
            });

            // Get all points within the radius for this image
            let mut image_result = self.within_impl(image, radius, distance).map(|evaluated| {
                evaluated
                    .into_vec()
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })?;

            canonical_result.append(&mut image_result)
        }

        // Now sort
        canonical_result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(canonical_result)
    }

    /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. Faster than within()
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within_unsorted<F>(
        &self,
        point: &[A; K],
        radius: A,
        distance: &F,
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.within_impl(point, radius, distance)
            .map(|evaluated| evaluated.into_vec().into_iter().map(Into::into).collect())
    }

        /// Queries the tree to find all elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned sorted nearest-first. Obeys periodic
    /// boundary conditions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// const PERIODIC: [f64; 3] = [10.0, 10.0, 10.0];
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let within = tree.within_unsorted_periodic(&[1.0, 2.0, 5.0], 10f64, &squared_euclidean, &PERIODIC)?;
    ///
    /// assert_eq!(within.len(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn within_unsorted_periodic<F>(
        &self,
        point: &[A; K],
        radius: A,
        distance: &F,
        periodic: &[A; K],
    ) -> Result<Vec<(A, &T)>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        let mut canonical_result: Vec<(A, &T)> = self.within_impl(point, radius, distance).map(|evaluated| {
            evaluated
                .into_vec()
                .into_iter()
                .map(Into::into)
                .collect()
        })?;


        // Find closest dist2 to every side
        let mut closest_side_dist2: [A; K] = [A::zero(); K];
        for side in 0..K {

            // Do a single index here. This is equal to distance to lower side
            let query_component: A = point[side];

            // Get distance to upper half
            let upper = periodic[side] - query_component;

            // !negative includes zero
            debug_assert!(!upper.is_negative()); 
            debug_assert!(!query_component.is_negative());

            // Choose lesser of two and then square
            closest_side_dist2[side] = upper.min(query_component).powi(2);
        }

        // Find which images we need to check.
        // Initialize vector with canonical image (which we will remove later)
        let mut images_to_check = Vec::with_capacity(2_usize.pow(K as u32)-1);
        for image in 1..2_usize.pow(K as u32) {
            
            // Closest image in the form of bool array
            let closest_image = (0..K)
                .map(|idx| ((image / 2_usize.pow(idx as u32)) % 2) == 1);

            // Find distance to corresponding side, edge, vertex or other higher dimensional equivalent
            let dist_to_side_edge_or_other: A = closest_image
                .clone()
                .enumerate()
                .flat_map(|(side, flag)| if flag {
                    
                    // Get minimum of dist2 to lower and upper side
                    Some(closest_side_dist2[side])
                } else { None })
                .fold(A::zero(), |acc, x| acc + x);

            if dist_to_side_edge_or_other < radius {

                let mut image_to_check = point.clone();
                
                for (idx, flag) in closest_image.enumerate() {

                    // If moving image along this dimension
                    if flag {
                        // Do a single index here. This is equal to distance to lower side
                        let query_component: A = point[idx];
                        // Single index here as well
                        let periodic_component = periodic[idx];

                        if query_component < periodic_component / A::from(2_u8).unwrap() {
                            // Add if in lower half of box
                            image_to_check[idx] = query_component + periodic_component;
                        } else {
                            // Subtract if in upper half of box
                            image_to_check[idx] = query_component - periodic_component;
                        }
                        
                    }
                }

                images_to_check.push(image_to_check);
            }
        }

        // Then check all images
        for image in &images_to_check {

            let mut image_pending = Vec::with_capacity(16);

            image_pending.push(HeapElement {
                distance: A::zero(),
                element: self,
            });

            // Get all points within the radius for this image
            let mut image_result = self.within_impl(image, radius, distance).map(|evaluated| {
                evaluated
                    .into_vec()
                    .into_iter()
                    .map(Into::into)
                    .collect()
            })?;

            canonical_result.append(&mut image_result)
        }

        Ok(canonical_result)
    }

    /// Queries the tree to find the best `n` elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 1)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let best_n_within = tree.best_n_within(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean)?;
    ///
    /// assert_eq!(best_n_within[0], 1);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn best_n_within<F>(
        &self,
        point: &[A; K],
        radius: A,
        max_qty: usize,
        distance: &F,
    ) -> Result<Vec<T>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
        T: Copy + Ord,
    {
        if self.size == 0 {
            return Ok(vec![]);
        }

        self.check_point(point)?;

        let mut pending = Vec::with_capacity(max_qty);
        let mut evaluated = BinaryHeap::<T>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() {
            self.best_n_within_step(
                point,
                self.size,
                max_qty,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        Ok(evaluated.into_vec().into_iter().collect())
    }

    /// Queries the tree to find the best `n` elements within `radius` of `point`, using the specified
    /// distance metric function. Results are returned in arbitrary order. 'Best' is determined by
    /// performing a comparison of the elements using < (ie, std::ord::lt). Returns an iterator.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 1)?;
    /// tree.add(&[200.0, 300.0, 600.0], 102)?;
    ///
    /// let mut best_n_within_iter = tree.best_n_within_into_iter(&[1.0, 2.0, 5.0], 10f64, 1, &squared_euclidean);
    /// let first = best_n_within_iter.next().unwrap();
    ///
    /// assert_eq!(first, 1);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn best_n_within_into_iter<F>(
        &self,
        point: &[A; K],
        radius: A,
        max_qty: usize,
        distance: &F,
    ) -> impl Iterator<Item = T>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
        T: Copy + Ord,
    {
        // if let Err(err) = self.check_point(point) {
        //     return Err(err);
        // }
        // if self.size == 0 {
        //     return std::iter::empty::<T>();
        // }

        let mut pending = Vec::with_capacity(max_qty);
        let mut evaluated = BinaryHeap::<T>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        while !pending.is_empty() {
            self.best_n_within_step(
                point,
                self.size,
                max_qty,
                radius,
                distance,
                &mut pending,
                &mut evaluated,
            );
        }

        evaluated.into_iter()
    }

    fn best_n_within_step<'b, F>(
        &self,
        point: &[A; K],
        _num: usize,
        max_qty: usize,
        max_dist: A,
        distance: &F,
        pending: &mut Vec<HeapElement<A, &'b Self>>,
        evaluated: &mut BinaryHeap<T>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
        T: Copy + Ord,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        <KdTree<A, T, K>>::populate_pending(point, max_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: self.get_distance(point, p, distance),
                    element: d,
                });

                for element in iter {
                    if element <= max_dist {
                        if evaluated.len() < max_qty {
                            evaluated.push(*element.element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element.element < &top {
                                *top = *element.element;
                            }
                        }
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn nearest_step<'b, F>(
        &self,
        point: &[A; K],
        num: usize,
        max_dist: A,
        distance: &F,
        pending: &mut BinaryHeap<HeapElement<A, &'b Self>>,
        evaluated: &mut BinaryHeap<HeapElement<A, &'b T>>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        <KdTree<A, T, K>>::populate_pending(point, max_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: self.get_distance(point, p, distance),
                    element: d,
                });

                for element in iter {
                    if element <= max_dist {
                        if evaluated.len() < num {
                            evaluated.push(element);
                        } else {
                            let mut top = evaluated.peek_mut().unwrap();
                            if element < *top {
                                *top = element;
                            }
                        }
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn get_distance<F>(
        &self,
        a: &[A; K],
        b: &[A; K],
        distance: &F,
    ) -> A
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        get_distance(a, b, distance, self.periodic)
    }

    fn nearest_one_step<'b, F>(
        &self,
        point: &[A; K],
        distance: &F,
        pending: &mut Vec<HeapElement<A, &'b Self>>,
        best_dist: &mut A,
        best_elem: &mut Option<&'b T>,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        let curr = &mut &*pending.pop().unwrap().element;
        let evaluated_dist = *best_dist;
        <KdTree<A, T, K>>::populate_pending(point, evaluated_dist, distance, pending, curr);

        match &curr.content {
            Node::Leaf { points, bucket, .. } => {
                let points = points.iter();
                let bucket = bucket.iter();
                let iter = points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: self.get_distance(point, p, distance),
                    element: d,
                });

                for element in iter {
                    if best_elem.is_none() || element < *best_dist {
                        *best_elem = Some(element.element);
                        *best_dist = element.distance;
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn populate_pending<'a, F>(
        point: &[A; K],
        max_dist: A,
        distance: &F,
        pending: &mut impl Stack<HeapElement<A, &'a Self>>,
        curr: &mut &'a Self,
    ) where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        while let Node::Stem { left, right, .. } = &curr.content {
            let candidate;
            if curr.belongs_in_left(point) {
                candidate = right;
                *curr = left;
            } else {
                candidate = left;
                *curr = right;
            };

            let candidate_to_space = util::distance_to_space(
                    point,
                    &candidate.min_bounds,
                    &candidate.max_bounds,
                    distance,
                );

            if candidate_to_space <= max_dist {
                pending.stack_push(HeapElement {
                    distance: candidate_to_space * -A::one(),
                    element: &**candidate,
                });
            }
        }
    }

    /// Returns an iterator over all elements in the tree, sorted nearest-first to the query point.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    /// use kiddo::distance::squared_euclidean;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[2.0, 3.0, 6.0], 101)?;
    ///
    /// let mut nearest_iter = tree.iter_nearest(&[1.0, 2.0, 5.1], &squared_euclidean)?;
    ///
    /// let nearest_first = nearest_iter.next().unwrap();
    ///
    /// assert!((nearest_first.0 - 0.01f64).abs() < f64::EPSILON);
    /// assert_eq!(*nearest_first.1, 100);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn iter_nearest<'a, 'b, F>(
        &'b self,
        point: &'a [A; K],
        distance: &'a F,
    ) -> Result<NearestIter<'a, 'b, A, T, F, K>, ErrorKind>
    where
        F: Fn(&[A; K], &[A; K]) -> A,
    {
        self.check_point(point)?;

        let mut pending = BinaryHeap::new();
        let evaluated = BinaryHeap::<HeapElement<A, &T>>::new();

        pending.push(HeapElement {
            distance: A::zero(),
            element: self,
        });

        Ok(NearestIter {
            point,
            pending,
            evaluated,
            distance,
            periodic: self.periodic
        })
    }

    /// Add an element to the tree. The first argument specifies the location in kd space
    /// at which the element is located. The second argument is the data associated with
    /// that point in space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kiddo::KdTree;
    ///
    /// let mut tree: KdTree<f64, usize, 3> = KdTree::new();
    ///
    /// tree.add(&[1.0, 2.0, 5.0], 100)?;
    /// tree.add(&[1.1, 2.1, 5.1], 101)?;
    ///
    /// assert_eq!(tree.size(), 2);
    /// # Ok::<(), kiddo::ErrorKind>(())
    /// ```
    pub fn add(&mut self, point: &[A; K], data: T) -> Result<(), ErrorKind> {
        self.check_point(point)?;
        self.add_unchecked(point, data)
    }

    fn add_unchecked(&mut self, point: &[A; K], data: T) -> Result<(), ErrorKind> {
        let res = match &mut self.content {
            Node::Leaf { .. } => {
                self.add_to_bucket(point, data);
                return Ok(());
            }

            Node::Stem {
                ref mut left,
                ref mut right,
                split_dimension,
                split_value,
            } => {
                if point[*split_dimension as usize] < *split_value {
                    // belongs_in_left
                    left.add_unchecked(point, data)
                } else {
                    right.add_unchecked(point, data)
                }
            }
        };

        self.extend(point);
        self.size += 1;

        res
    }

    fn add_to_bucket(&mut self, point: &[A; K], data: T) {
        self.extend(point);
        let cap;
        match &mut self.content {
            Node::Leaf {
                ref mut points,
                ref mut bucket,
                capacity,
            } => {
                points.push(*point);
                bucket.push(data);
                cap = *capacity;
            }
            Node::Stem { .. } => unreachable!(),
        }

        self.size += 1;
        if self.size > cap {
            self.split();
        }
    }

    pub fn remove(&mut self, point: &[A; K], data: &T) -> Result<usize, ErrorKind> {
        let mut removed = 0;
        self.check_point(point)?;

        match &mut self.content {
            Node::Leaf {
                ref mut points,
                ref mut bucket,
                ..
            } => {
                let mut p_index = 0;
                while p_index < self.size {
                    if &points[p_index] == point && &bucket[p_index] == data {
                        points.swap_remove(p_index);
                        bucket.swap_remove(p_index);
                        removed += 1;
                        self.size -= 1;
                    } else {
                        p_index += 1;
                    }
                }
            }
            Node::Stem {
                ref mut left,
                ref mut right,
                ..
            } => {
                let right_removed = right.remove(point, data)?;
                if right_removed > 0 {
                    self.size -= right_removed;
                    removed += right_removed;
                }

                let left_removed = left.remove(point, data)?;
                if left_removed > 0 {
                    self.size -= left_removed;
                    removed += left_removed;
                }
            }
        }

        Ok(removed)
    }

    fn split(&mut self) {
        match &mut self.content {
            Node::Leaf {
                ref mut bucket,
                ref mut points,
                capacity,
                ..
            } => {
                let mut split_dimension: Option<usize> = None;
                let mut max = A::zero();
                for dim in 0..K {
                    let diff = self.max_bounds[dim] - self.min_bounds[dim];
                    if !diff.is_nan() && diff > max {
                        max = diff;
                        split_dimension = Some(dim);
                    }
                }

                if let Some(split_dimension) = split_dimension {
                    let min = self.min_bounds[split_dimension];
                    let max = self.max_bounds[split_dimension];
                    let split_value = min + (max - min) / A::from(2.0_f64).unwrap();

                    let mut left;
                    let mut right;
                    if self.periodic.is_some() {
                       left = Box::new(KdTree::periodic_with_per_node_capacity(*capacity, self.periodic.unwrap()).unwrap());
                       right = Box::new(KdTree::periodic_with_per_node_capacity(*capacity, self.periodic.unwrap()).unwrap());
                    } else {
                       left = Box::new(KdTree::with_per_node_capacity(*capacity).unwrap());
                       right = Box::new(KdTree::with_per_node_capacity(*capacity).unwrap());
                    }


                    while !points.is_empty() {
                        let point = points.swap_remove(0);
                        let data = bucket.swap_remove(0);
                        if point[split_dimension] < split_value {
                            // belongs_in_left
                            left.add_to_bucket(&point, data);
                        } else {
                            right.add_to_bucket(&point, data);
                        }
                    }

                    self.content = Node::Stem {
                        left,
                        right,
                        split_value,
                        split_dimension: split_dimension as u8,
                    }
                }
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    fn belongs_in_left(&self, point: &[A; K]) -> bool {
        match &self.content {
            Node::Stem {
                ref split_dimension,
                ref split_value,
                ..
            } => point[*split_dimension as usize] < *split_value,
            Node::Leaf { .. } => unreachable!(),
        }
    }

    fn extend(&mut self, point: &[A; K]) {
        let min = self.min_bounds.iter_mut();
        let max = self.max_bounds.iter_mut();
        for ((l, h), v) in min.zip(max).zip(point.iter()) {
            if v < l {
                *l = *v
            }
            if v > h {
                *h = *v
            }
        }
    }

    fn check_point(&self, point: &[A; K]) -> Result<(), ErrorKind> {

        // First check that point is finite
        if !point.iter().all(|n| n.is_finite()) {
            return Err(ErrorKind::NonFiniteCoordinate)
        }

        // Then check that point is in the bounds when periodic BCs are on
        if self.periodic.is_some() {

            // Used several times, so unwrap once here.
            let periodic = self.periodic.unwrap();

            // Check if any component is above or below the bound specified in `periodic`
            let (above_bound, below_bound) = periodic
                .iter()
                .fold((false, false), |mut acc, &x| {
                    for idx in 0..K {
                        acc = (acc.0 || x > periodic[idx], acc.1 || x < periodic[idx]);
                    }
                    acc
                });

            // If so, then return 
            if above_bound || below_bound {
                return Err(ErrorKind::PeriodicOutOfBounds)
            }
        }

        Ok(())
    }
}

pub struct NearestIter<
    'a,
    'b,
    A: 'a + 'b + Float,
    T: 'b + PartialEq,
    F: 'a + Fn(&[A; K], &[A; K]) -> A,
    const K: usize,
> {
    point: &'a [A; K],
    pending: BinaryHeap<HeapElement<A, &'b KdTree<A, T, K>>>,
    evaluated: BinaryHeap<HeapElement<A, &'b T>>,
    distance: &'a F,
    periodic: Option<[A; K]>,
}

impl<'a, 'b, A: Float + Zero + One + Signed, T: 'b, F: 'a, const K: usize> Iterator
    for NearestIter<'a, 'b, A, T, F, K>
where
    F: Fn(&[A; K], &[A; K]) -> A,
    T: PartialEq,
{
    type Item = (A, &'b T);
    fn next(&mut self) -> Option<(A, &'b T)> {
        use util::distance_to_space;

        let distance = self.distance;
        let point = self.point;
        while !self.pending.is_empty()
            && (self.evaluated.peek().map_or(A::infinity(), |x| -x.distance)
                >= -self.pending.peek().unwrap().distance)
        {
            let mut curr = &*self.pending.pop().unwrap().element;
            while let Node::Stem { left, right, .. } = &curr.content {
                let candidate;
                if curr.belongs_in_left(point) {
                    candidate = right;
                    curr = left;
                } else {
                    candidate = left;
                    curr = right;
                };
                self.pending.push(HeapElement {
                    distance: 
                        -distance_to_space(
                            point,
                            &candidate.min_bounds,
                            &candidate.max_bounds,
                            distance,
                        ),
                    element: &**candidate,
                });
            }

            // Local clone of periodic to satisfy borrow checker before mut borrow
            let periodic = self.periodic.clone();
            match &curr.content {
                Node::Leaf { points, bucket, .. } => {
                    let points = points.iter();
                    let bucket = bucket.iter();

                    self.evaluated
                        .extend(points.zip(bucket).map(|(p, d)| HeapElement {
                            distance: -get_distance(point, p, distance, periodic),
                            element: d,
                        }));
                }
                Node::Stem { .. } => unreachable!(),
            }
        }
        self.evaluated.pop().map(|x| (-x.distance, x.element))
    }
}

pub fn get_distance<'a, 'b, A, F, const K: usize>(
    a: &[A; K],
    b: &[A; K],
    distance: &F,
    periodic: Option<[A; K]>,
) -> A
where
    A: 'a + 'b + Float,
    F: Fn(&[A; K], &[A; K]) -> A,
{
    // If not using periodic boundary conditions, just calculate and return distance
    if periodic.is_none() {

        return distance(a, b)

    
    // Otherwise, calculate the minimum distance from all mirror images
    } else {
        
        // Initialize min to max possible distance
        let mut min: A = periodic
            .unwrap()
            .iter()
            .fold(
                A::zero(),
                |acc, &x| acc + x*x,
            );

        // Calculate distance for every image
        for image_idx in 0..3_i32.pow(K as u32) {

            // Initialize current_image template
            let mut current_image: [i32; K] = [0; K];

            // Calculate current image
            for idx in 0..K {
                current_image[idx] = (( image_idx / 3_i32.pow(idx as u32)) % 3) - 1;
            }

            // Construct current image position
            let mut new_a: [A; K] = a.clone();
            for idx in 0..K {
                new_a[idx] = new_a[idx] + A::from(current_image[idx]).unwrap()*periodic.unwrap()[idx];
            }

            // Calculate distance for this image
            let image_distance = distance(&new_a, b);

            // Compare with current min
            min = min.min(image_distance);
        }

        min
    }
}

impl std::error::Error for ErrorKind {}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let reason = match *self {
            ErrorKind::PeriodicOutOfBounds => "out-of-bounds when using periodic boundary conditions",
            ErrorKind::NonFiniteCoordinate => "non-finite coordinate",
            ErrorKind::ZeroCapacity => "zero capacity",
            ErrorKind::Empty => "invalid operation on empty tree",
        };
        write!(f, "KdTree error: {}", reason)
    }
}

#[cfg(test)]
mod tests {
    extern crate rand;
    use super::KdTree;
    use super::Node;

    fn random_point() -> ([f64; 2], i32) {
        rand::random::<([f64; 2], i32)>()
    }

    #[test]
    fn it_has_default_capacity() {
        let tree: KdTree<f64, i32, 2> = KdTree::new();
        match &tree.content {
            Node::Leaf { capacity, .. } => {
                assert_eq!(*capacity, 2_usize.pow(4));
            }
            Node::Stem { .. } => unreachable!(),
        }
    }

    #[test]
    fn it_can_be_cloned() {
        let mut tree: KdTree<f64, i32, 2> = KdTree::new();
        let (pos, data) = random_point();
        tree.add(&pos, data).unwrap();
        let mut cloned_tree = tree.clone();
        cloned_tree.add(&pos, data).unwrap();
        assert_eq!(tree.size(), 1);
        assert_eq!(cloned_tree.size(), 2);
    }

    #[test]
    fn it_holds_on_to_its_capacity_before_splitting() {
        let mut tree: KdTree<f64, i32, 2> = KdTree::new();
        let capacity = 2_usize.pow(4);
        for _ in 0..capacity {
            let (pos, data) = random_point();
            tree.add(&pos, data).unwrap();
        }
        assert_eq!(tree.size, capacity);
        assert_eq!(tree.size(), capacity);
        assert!(tree.is_leaf());
        {
            let (pos, data) = random_point();
            tree.add(&pos, data).unwrap();
        }
        assert_eq!(tree.size, capacity + 1);
        assert_eq!(tree.size(), capacity + 1);
        assert!(!tree.is_leaf());
    }
}
