# Mops

<u>M</u>athematical <u>op</u>eration<u>s</u> to extract all the performance juice from your hardware!

## Getting the code

```bash
git clone https://github.com/lab-cosmo/mops
cd mops

# Builds the code and run all tests
tox

# Installs the Python package
pip install .
```


## Implemented operations

Some common motifs of vector, matrix and tensor operations that appear in
science and engineering are implemented here for CPUs and GPUs. These
include:

### 1. Homogeneous Polynomial Evaluation

#### Mathematical notation

$$ O_i = \sum_{j=1}^J C_j \prod_{k=1}^K A_{jP_{jk}} $$

#### Inputs

- $A_i$ is a dense 3D tensor of floats, expected to be very large in one
  dimension ($N$), much smaller in the other two ($F \times L_i$)

- $W$ is a vector of float multipliers of size $Q$.

- $M_1, M_2, ...$ are vectors of integers of size $Q$ containing indices that point to slices in $A$

- $M_O$ is a vector of integers of size $Q$ containing indices that point to slices in $E$

#### Output

$E$ is a dense 3D tensor of floats, expected to be very large in one dimension
($N$), much smaller in the other two ($F \times L_O$). It contains the
accumulated tensor product.

#### Calculation

Each entry in $M_O$, $M_1$, $M_2$, etc point to slices in $E$, $A_1$, $A_2$, etc
respectively.

For a given set of entries from the same index $i$ in the $M$ arrays and $W$
array, we add the following to each slice $E[:,:, M_O[i]]$:

$$ W[i] * ( A_1[:, :, M_1[i]] \odot A_2[:, :, M_2[i]] \odot \dots ) $$

Here, $\odot$ implies element-wise matrix multiplication between two matrices of
identical shape, in this case the slices of $A_1$, $A_2$, etc.

```python
for j in range(J):
    O[:] += C[j] * A[:, P_1[j, 1]] * A[:, P_2[j, 2]] * ...
```

### 2. Sparse accumulation of Products

#### Mathematical notation

$$ O_{iP_k^O} = \sum_{k \in \{k'|P^O_{k'}=P^O_k\}} C_k A_{iP_k^A} B_{iP_k^B} $$

#### Calculation

```python
for k in range(K):
    O[:, P_O[k]] += C[k] * A[:, P_A[k]] * A[:, P_B[k]]
```

### 3. Outer Product Scatter-Add

#### Math notation

$$ O_{ikl} = \sum_{j=1}^J A_{jk} B_{jl} \delta_{iP_j} \hspace{1cm} \mathrm{or} \hspace{1cm} O_{ikl} = \sum_{j \in \{j'|P_{j'}=i\}} A_{jk} B_{jl} $$

#### Calculation

#### Inputs

- $A$ is a dense matrix of floats, expected to be very large in one dimension
  ($N$), but finite/fixed in the the other ($F$).

- $B$ is a dense matrix of floats, expected to be very large in one dimension
  ($N$), much smaller in the other ($L$).

- $J$ is a very large vector of integers (of size $N$) where each element is
  linked to the corresponding row in the matrices $K$ and $S$ and the value is
  linked to first dimension of the output tensor $A$.

#### Output

$C$ is a dense 3D tensor of floats, expected to be very large in one dimension
($M$), much smaller in the other two ($F \times L$)

#### Calculation

Each row of the input matrices $A$ and $B$ can be treated like vectors. For each
pair of vectors of size $F$ and $L$ respectively, we calculate the outer product
which results in a (small) matrix of size $F \times L$.

The output tensor $C$ can be treated like a stack of $M$ small matrices. Each
matrix computed from the outer product of each row is added to the matrix at the
index mentioned in corresponding element of the vector $J$.

```python
for j in range(J):
    O[P[j], :, :] += A[j, :, None] * B[j, None, :]
```

### 4. Outer product Scatter-Add with Weights

#### Math notation

$$ O_{imk} = \sum_{e \in \{e'|I_{e'}=i\}} A_{em} R_{ek} X_{{J_e}k} $$

#### Calculation

```python
for e in range(E):
    O[I[e], :, :] += A[e, :, None] * R[e, None, :] * X[J[e], None, :]
```

### 5. Sparse Accumulation Scatter-Add with Weights

#### Math notation

$$ O_{i{m_3}k} = \sum_{e \in \{e'|I_{e'}=i\}} R_{ek} \sum_{n \in \{n'|M^3_{n'}=m_3\}} C_n A_{e{M_n^1}} X_{{J_e}{M_n^2}k} $$

#### Calculation

```python
for e in range(E):
    for n in range(N):
        O[I[e], M_3[n], :] += R[e, :] * C[n] * A[e, M_1[n]] * X[J[e], M_2[n], :]
```
