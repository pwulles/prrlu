# PRRLU

Partial Rank-Revealing LU (PRRLU) decomposition for matrices stored in the  
[`mdarray`](https://crates.io/crates/mdarray) ecosystem.

This crate provides a stable and efficient implementation of PRRLU, relying on  
the backend traits from [`mdarray_linalg`](https://crates.io/crates/mdarray_linalg).  
Two backends are available out of the box: a hand-written **Naive** backend and a  
high-performance **Blas** backend.

With BLAS, the algorithm achieves significantly better performance than the  
naive loop-based implementation, while still remaining fully compatible with any  
custom backend implementing the required `Outer` and `Argmax` traits.

---

## Features

- Stable implementation of Partial Rank-Revealing LU decomposition.
- Works directly on `mdarray` matrices.
- Backend-agnostic: any backend implementing the required traits can be used.
- Includes both a Naive backend and a high-performance BLAS backend.
- Optional epsilon tolerance and optional target rank.

---

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
prrlu = "0.1"
mdarray = "0.7.1"
mdarray_linalg = "0.2"
mdarray_linalg_blas = "0.2" # optional, for BLAS backend
```

---

## Example

```rust
use mdarray::tensor;
use prrlu::{PRRLU, PRRLUDecomp};

let mut a = tensor![[1., 2.], [3., 4.]];
let decomp = PRRLU::new(&mut a).decompose();
let PRRLUDecomp { p, l, u, q, rank } = decomp; // a ≈ p^T * l * u * q^T
println!("rank: {}", rank);
```

--- 
## License

Dual-licensed under Apache 2.0 and MIT to ensure compatibility with the
Rust ecosystem. See LICENSE.md for details.
