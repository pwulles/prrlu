use mdarray_linalg::prelude::{Argmax, MatMul, MatMulBuilder, Outer};
use mdarray_linalg::testing::common::{random_matrix, rank_k_matrix};
use mdarray_linalg::{Naive, assert_matrix_eq};

use mdarray_linalg_blas::Blas;

use approx::assert_relative_eq;

use mdarray::DTensor;
use prrlu::{PRRLU, PRRLUDecomp};

#[test]
fn rank_deficient() {
    test_rank_deficient(Naive);
    test_rank_deficient(Blas);
}

#[test]
fn full_rank() {
    test_full_rank(Naive);
    test_full_rank(Blas);
}

#[test]
fn rectangular() {
    test_rectangular(Naive);
    test_rectangular(Blas);
}

#[test]
fn hilbert_matrix() {
    test_hilbert_matrix(Naive);
    test_hilbert_matrix(Blas);
}

/// Reconstruct matrix from PRRLU decomposition: A = P^-1 * L * U * Q^-1
fn reconstruct_from_prrlu(decomp: &PRRLUDecomp<f64>) -> DTensor<f64, 2> {
    let PRRLUDecomp { p, l, u, q, .. } = decomp;

    let p_inv = p.transpose();
    let q_inv = q.transpose();

    let temp1 = Naive.matmul(l, u).eval();
    let temp2 = Naive.matmul(&p_inv, &temp1).eval();
    Naive.matmul(&temp2, &q_inv).eval()
}

pub fn test_rank_deficient(bd: impl Outer<f64> + Argmax<f64>) {
    let n = 10;
    let m = 5;
    let k = 2; // rank

    // Generate rank-k matrix
    let original = rank_k_matrix(n, m, k);
    let mut a = original.clone();

    let decomp = PRRLU::new(&mut a).backend(bd).decompose();
    let reconstructed = reconstruct_from_prrlu(&decomp);

    println!("{:?}", decomp.u);

    assert_eq!(decomp.rank, k);
    assert_matrix_eq!(original, reconstructed);
}

pub fn test_full_rank(bd: impl Outer<f64> + Argmax<f64>) {
    let n = 10;
    let m = 10;

    // Generate a well-conditioned full rank matrix
    let original = random_matrix(m, n);
    let mut a = original.clone();

    let decomp = PRRLU::new(&mut a).backend(bd).decompose();
    let reconstructed = reconstruct_from_prrlu(&decomp);

    assert_eq!(decomp.rank, n);
    assert_matrix_eq!(original, reconstructed);
}

pub fn test_rectangular(bd: impl Outer<f64> + Argmax<f64>) {
    let n = 4;
    let m = 6;
    let k = 2;

    // Test with more columns than rows
    let original = rank_k_matrix(n, m, k);
    let mut a = original.clone();

    let decomp = PRRLU::new(&mut a).backend(bd).decompose();
    let reconstructed = reconstruct_from_prrlu(&decomp);

    println!("{:?}", decomp.u);

    assert_eq!(decomp.rank, k);
    assert_matrix_eq!(original, reconstructed);
}

// Generate Hilbert matrix H_ij = 1/(i+j+1)
fn gen_hilbert_matrix(n: usize) -> DTensor<f64, 2> {
    DTensor::<f64, 2>::from_fn([n, n], |idx| 1.0 / (idx[0] + idx[1] + 1) as f64)
}

pub fn test_hilbert_matrix(bd: impl Outer<f64> + Argmax<f64>) {
    let n = 20;

    let original = gen_hilbert_matrix(n);
    let mut a = original.clone();

    let decomp = PRRLU::new(&mut a).backend(bd).decompose();
    let reconstructed = reconstruct_from_prrlu(&decomp);

    assert_matrix_eq!(original, reconstructed);
}
