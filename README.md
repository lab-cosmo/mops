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
include, in alphabetical order:

### 1. Homogeneous Polynomial Evaluation

$$ O_i = \sum_{j=1}^J C_j \sum_{k=1}^K A_{j, P_{jk}} \,\,\,\, \mathrm{for} \,\, i=1, \dots, I$$

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

### 2. Sparse accumulation

$$ O_{i, P_k^O} = \sum_{k=1}^K C_k A_{i, P_k^A} B_{i, P_k^B} \,\,\,\, \mathrm{for} \,\, i=1, \dots, I$$

### 3. Sparse Accumulation Scatter-Add 

$$ O_{i, P_k^O, l} = \sum_{k=1}^K C_k \sum_{j \in \mathcal{N}(i)} A_{j, P_k^A, l} B_{j, P_k^B, l} \,\,\,\, \mathrm{for} \,\, l=1, \dots, L$$

### 4. Outer Product Scatter-Add

$$ O_{ikl} = \sum_{j \in \mathcal{N}(i)} A_{jk} B_{jl} \,\,\,\, \mathrm{for} \,\, k=1, \dots, K \,\,\,\, \mathrm{for} \,\, l=1, \dots, L$$

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

In index notation,
$$C[i,:, :] = \sum_{j \in J[i]} A[j, :] \otimes B[j, :]  $$

