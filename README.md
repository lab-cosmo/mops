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


## Planned operations

Some common motifs of vector, matrix and tensor operations that appear in
science and engineering are planned to be implemented here for CPUs and GPUs.
These include:

### 1. Homogeneous Polynomial Evaluation

#### Mathematical notation

$$ O_i = \sum_{j=1}^J C_j \prod_{k=1}^K A_{iP_{jk}} $$

#### Inputs

- $A$ is a 2D array of floats, of size $I \times N_{A,2}$. It contains the individual factors in the monomials that make up the polynomial. 

- $C$ is a vector of float multipliers of size $J$. They represent the coefficients of each monomial in the polynomial, so that $J$ is the number of monomials in the polynomial.

- $P$ is a 2D array of integers which represents the positions of the individual factors for each monomial in the second dimension of the $A$ array. In particular, the $k$-th factor of monomial $j$ will be found in the $P_{jk}$-th position of the second dimension of $A$.

#### Output

$O$ is a dense 1D array of floats, which only contains a batch dimension of size $I$.

#### Calculation

The calculation consists in a batched evaluation of homogeneous polynomials of degree $K$, where the monomials are given by $C[j] * A[:, P_1[j, 1]] * A[:, P_2[j, 2]] * \dots$, as follows:

```python
for j in range(J):
    O[:] += C[j] * A[:, P_1[j, 1]] * A[:, P_2[j, 2]] * ...
```

### 2. Sparse Accumulation of Products

#### Mathematical notation

$$ O_{iP_k^O} = \sum_{k \in \{k'|P^O_{k'}=P^O_k\}} C_k A_{iP_k^A} B_{iP_k^B} $$

#### Inputs

- $A$ and $B$ are 2D arrays of floats whose first dimension is a batch dimension that has the same size for both. 

- $C$ is a 1D array of floats which contains the weights of the products of elements of $A$ and $B$ to be accumulated.

- $P^A$, $P^B$ are 1D arrays fo integers of the same size which contain the positions along the second dimensions of $A$ and $B$, respectively, of the factors that constitute the products.

- $P^O$ is a 1D array of integers of the same length as $P^A$ and $P^B$ which contains the positions in the second dimension of the output tensor where the different products of $A$ and $B$ are accumulated.

#### Output

$O$ is a 2D array of floats where the first dimension is a batch dimension (the same as in $A$ and $B$) and the second dimension contains the scattered products of $A$ and $B$.

#### Calculation

The weighted products of $A$ and $B$ are accumulated into $O$ as follows:

```python
for k in range(K):
    O[:, P_O[k]] += C[k] * A[:, P_A[k]] * B[:, P_B[k]]
```

### 3. Outer Product Scatter-Add

#### Math notation

$$ O_{ikl} = \sum_{j=1}^J A_{jk} B_{jl} \delta_{iP_j} \hspace{1cm} \mathrm{or} \hspace{1cm} O_{ikl} = \sum_{j \in \{j'|P_{j'}=i\}} A_{jk} B_{jl} $$

#### Inputs

- $A$ is a dense matrix of floats, expected to be large in one dimension
  (size $J$), and smaller in the the other (size $K$).

- $B$ is a dense matrix of floats, expected to be large in one dimension
  (size $J$), and smaller in the the other (size $L$).

- $P$ is a large vector of integers (of size $J$) which maps the dimension $j$ of $A$ and $B$ into the dimension $i$ of $O$. In other words, it contains the position within $O$ where each $AB$ product needs to be summed.

- $n_O$ is the size of the output array along its first dimension. It must be grater or equal than the larger element in $P$ plus one. 

#### Output

$O$ is a 3D array of floats of dimensions $I \times K \times L$, which contains the accumulated products of the elements of $A$ and $B$.

#### Calculation

For each $j$, an outer product of $A[j, :]$ and $B[j, :]$ is calculated, and it is summed to $O[P[j], :, :]$:

```python
for j in range(J):
    O[P[j], :, :] += A[j, :, None] * B[j, None, :]
```

### 4. Outer product Scatter-Add with Weights

#### Math notation

$$ O_{imk} = \sum_{e \in \{e'|I_{e'}=i\}} A_{em} R_{ek} X_{{J_e}k} $$

#### Inputs

- $A$ is a 2D array of floats
- $R$ is a 2D array of floats
- $X$ is a 2D array of floats
- $I$ is a 1D array of integers 
- $J$ is a 1D array of integers 

#### Outputs

- $O$ is a 3D array of floats

#### Calculation

```python
for e in range(E):
    O[I[e], :, :] += A[e, :, None] * R[e, None, :] * X[J[e], None, :]
```

### 5. Sparse Accumulation Scatter-Add with Weights

#### Math notation

$$ O_{i{m_3}k} = \sum_{e \in \{e'|I_{e'}=i\}} R_{ek} \sum_{n \in \{n'|M^3_{n'}=m_3\}} C_n A_{e{M_n^1}} X_{{J_e}{M_n^2}k} $$

#### Inputs

#### Inputs

- $A$ is a 2D array of floats
- $R$ is a 2D array of floats
- $X$ is a 3D array of floats
- $C$ is a 1D array of floats
- $I$ is a 1D array of integers 
- $J$ is a 1D array of integers 
- $M^1$ is a 1D array of integers 
- $M^2$ is a 1D array of integers 
- $M^3$ is a 1D array of integers 

#### Outputs

- $O$ is a 3D array of floats

#### Outputs

#### Calculation

```python
for e in range(E):
    for n in range(N):
        O[I[e], M_3[n], :] += R[e, :] * C[n] * A[e, M_1[n]] * X[J[e], M_2[n], :]
```
