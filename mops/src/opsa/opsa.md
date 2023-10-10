
# Outer Product Scatter-Add

## Inputs

$K$ is a dense matrix of floats, expected to be very large in one dimension ($N$), but finite/fixed in the the other ($F$).

$S$ is a dense matrix of floats, expected to be very large in one dimension ($N$), much smaller in the other ($L$).

$J$ is a very large vector of integers (of size $N$) where each element is linked to the corresponding row in the matrices $K$ and $S$ and the value is linked to first dimension of the output tensor $A$.

## Output

$A$ is a dense 3D tensor of floats, expected to be very large in one dimension ($N$), much smaller in the other two ($F \times L$)

## Calculation

Each row of the matrices $K$ and $S$ can be treated like vectors.
For each pair of vectors of size $F$ and $L$ respectively, we calculate the outer product which results in a (small) matrix of size $F \times L$.

The tensor $A$ can be treated like a stack of $N$ small matrices.
Each matrix computed from the outer product of each row is added to the matrix at the index mentioned in corresponding element of the vector $J$.

In index notation,
$$A[i,:, :] = \sum_{j \in J[i]} K[j, :] \otimes S[j, :]  $$

# Sparse Accumulation for Pre-Computed Tensors

## Inputs

$A_i$ is a dense 3D tensor of floats, expected to be very large in one dimension ($N$), much smaller in the other two ($F \times L_i$)

$W$ is a vector of float multipliers of size $Q$.

$M_1, M_2, ...$ are vectors of integers of size $Q$ containing indices that point to slices in $A$

$M_O$ is a vector of integers of size $Q$ containing indices that point to slices in $E$

## Output

$E$ is a dense 3D tensor of floats, expected to be very large in one dimension ($N$), much smaller in the other two ($F \times L_O$). It contains the accumulated tensor product.

## Calculation

Each entry in $M_O$, $M_1$, $M_2$, etc point to slices in $E$, $A_1$, $A_2$, etc respectively.

For a given set of entries from the same index $i$ in the $M$ arrays and $W$ array, we add the following to each slice $E[:,:, M_O[i]]$ 

$$ W[i] * ( A_1[:, :, M_1[i]] \odot A_2[:, :, M_2[i]] \odot \dots ) $$
