//! Partial rank-revealing LU decomposition utilities.
//!
//! # Examples
//!
//! Basic usage with default settings:
//! ```rust
//! use mdarray::tensor;
//! use prrlu::{PRRLU, PRRLUDecomp};
//!
//! let mut a = tensor![[1., 2.], [3., 4.]];
//! let decomp = PRRLU::new(&mut a).decompose();
//! let PRRLUDecomp { p, l, u, q, rank } = decomp; // a ≈ p^T * l * u * q^T
//! println!("rank: {}", decomp.rank);
//! ```
//!
//! With custom backend:
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg_blas::Blas;
//! use prrlu::PRRLU;
//!
//! let mut a = tensor![[1., 2.], [3., 4.]];
//! let decomp = PRRLU::new(&mut a).backend(Blas).decompose();
//! ```
//!
//! With custom epsilon tolerance:
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg_blas::Blas;
//! use prrlu::PRRLU;
//!
//! let mut a = tensor![[1., 2.], [3., 4.]];
//! let decomp = PRRLU::new(&mut a)
//!     .backend(Blas)
//!     .epsilon(1e-15)
//!     .decompose();
//! ```
//!
//! With target rank:
//! ```rust
//! use mdarray::tensor;
//! use mdarray_linalg_blas::Blas;
//! use prrlu::PRRLU;
//!
//! let mut a = tensor![[1., 2.], [3., 4.]];
//! let decomp = PRRLU::new(&mut a)
//!     .backend(Blas)
//!     .rank(2)
//!     .decompose();
//! ```

use mdarray::{DSlice, DTensor, Layout};
use mdarray_linalg::matvec::{Argmax, Outer, OuterBuilder};
use mdarray_linalg::{Naive, identity};
use num_traits::{FromPrimitive, Signed};

/// Defines the numeric requirements for types used in PRRLU computations.
pub trait PRRLUScalar:
    Copy
    + PartialOrd
    + Signed
    + std::ops::Sub<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::fmt::Debug
    + FromPrimitive
    + num_traits::Float
    + num_traits::FloatConst
    + std::convert::From<i8>
    + 'static
{
    fn default_epsilon() -> Self {
        Self::from_f64(1e-12).unwrap()
    }
}

impl<T> PRRLUScalar for T where
    T: Copy
        + PartialOrd
        + Signed
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Neg<Output = T>
        + std::fmt::Debug
        + FromPrimitive
        + num_traits::Float
        + num_traits::FloatConst
        + std::convert::From<i8>
        + 'static
{
}

/// Holds the results of a pivoted, rank-revealing LU decomposition,
/// including permutation matrices and the computed rank
pub struct PRRLUDecomp<T> {
    pub p: DTensor<T, 2>,
    pub l: DTensor<T, 2>,
    pub u: DTensor<T, 2>,
    pub q: DTensor<T, 2>,
    pub rank: usize,
}

/// Holds the parameters required to perform the PRRLU transformation.
pub struct PRRLU<'a, T: PRRLUScalar, L: Layout, B: Outer<T> + Argmax<T>> {
    matrix: &'a mut DSlice<T, 2, L>,
    bd: B,
    epsilon: Option<T>,
    target_rank: Option<usize>,
}

impl<'a, T: PRRLUScalar, L: Layout> PRRLU<'a, T, L, Naive> {
    pub fn new(matrix: &'a mut DSlice<T, 2, L>) -> Self {
        Self {
            matrix,
            bd: Naive,
            epsilon: None,
            target_rank: None,
        }
    }
}

impl<'a, T: PRRLUScalar, L: Layout, B: Outer<T> + Argmax<T>> PRRLU<'a, T, L, B> {
    pub fn backend<B2: Outer<T> + Argmax<T>>(self, bd: B2) -> PRRLU<'a, T, L, B2> {
        PRRLU {
            matrix: self.matrix,
            bd,
            epsilon: self.epsilon,
            target_rank: self.target_rank,
        }
    }

    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = Some(epsilon);
        self
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.target_rank = Some(rank);
        self
    }

    /// Compute full Partial Rank-Revealing LU decomposition
    ///
    /// Decomposes matrix `A` into the form: `A = P * L * U * Q` where
    /// `P`, `Q` are permutation matrices, `L` is unit lower
    /// triangular and `U` is unit upper triangular.
    ///
    /// Algorithm: iteratively selects the maximum element as pivot,
    /// permutes rows/columns to bring it to diagonal position,
    /// computes Schur complement for the `(n-1)×(n-1)` subblock,
    /// and repeats until completion or pivot falls below numerical precision.
    pub fn decompose(self) -> PRRLUDecomp<T> {
        let (m, n) = *self.matrix.shape();
        let mut p = identity::<T>(m);
        let mut q = identity::<T>(n);
        let mut l = identity::<T>(m);

        let k = self.target_rank.unwrap_or_else(|| m.max(n));
        let epsilon = self.epsilon.unwrap_or_else(T::default_epsilon);

        let rank = raw_prrlu(self.matrix, &mut p, &mut q, &mut l, k, epsilon, self.bd);

        PRRLUDecomp {
            p,
            l,
            u: self.matrix.as_ref().to_owned().into(),
            q,
            rank,
        }
    }
}

fn gaussian_elimination<T: PRRLUScalar, L: Layout>(
    work: &mut DSlice<T, 2, L>,
    lower: &mut DSlice<T, 2>,
    step: usize,
    pivot: T,
    bd: &impl Outer<T>,
) {
    let n = work.shape().0;
    if step < n - 1 {
        let multipliers: Vec<_> = work
            .view(step + 1.., step)
            .iter()
            .map(|x| *x / pivot)
            .collect();
        let wv: Vec<_> = work.view(step, step..).iter().copied().collect();
        lower
            .view_mut(step + 1.., step)
            .iter_mut()
            .zip(multipliers.iter())
            .for_each(|(l, m)| *l = *m);
        let md_multipliers = DTensor::<T, 1>::from(multipliers);
        let md_wv = DTensor::<T, 1>::from(wv);

        bd.outer(&md_multipliers, &md_wv)
            .scale(-T::one())
            .add_to_overwrite(&mut work.view_mut(step + 1.., step..))
    }
}

fn update_permutation_matrices<T>(
    p: &mut DSlice<T, 2>,
    q: &mut DSlice<T, 2>,
    lower: &mut DSlice<T, 2>,
    idx_pivot: &[usize],
    step: usize,
) {
    if idx_pivot[0] != step {
        p.swap_axis(0, step, idx_pivot[0]);
        if step > 0 {
            let mut lower_view = lower.view_mut(.., 0..step);
            lower_view.swap_axis(0, step, idx_pivot[0]);
        }
    }

    if idx_pivot[1] != step {
        q.swap_axis(1, step, idx_pivot[1]);
    }
}

fn raw_prrlu<T: PRRLUScalar, L: Layout>(
    a: &mut DSlice<T, 2, L>,
    p: &mut DSlice<T, 2>,
    q: &mut DSlice<T, 2>,
    lower: &mut DSlice<T, 2>,
    k: usize,
    epsilon: T,
    bd: impl Outer<T> + Argmax<T>,
) -> usize {
    let (m, n) = *a.shape();
    let max_steps = k.min(n.min(m));

    let mut idx_pivot = bd.argmax_abs(a).unwrap();

    for step in 0..max_steps {
        a.swap_axis(0, step, idx_pivot[0]);
        a.swap_axis(1, step, idx_pivot[1]);

        update_permutation_matrices(p, q, lower, &idx_pivot, step);

        let pivot_current = a[[step, step]];

        if is_pivot_too_small(pivot_current, epsilon) {
            return step;
        }

        gaussian_elimination(a, lower, step, pivot_current, &bd);
        idx_pivot = bd
            .argmax_abs(&a.view(step + 1.., step..))
            .unwrap_or(vec![step, step]);

        idx_pivot[0] += step + 1;
        idx_pivot[1] += step;
    }
    max_steps
}

#[inline]
fn is_pivot_too_small<T: PRRLUScalar>(pivot: T, epsilon: T) -> bool {
    pivot.abs() < epsilon
}
