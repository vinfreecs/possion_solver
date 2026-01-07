__global__ void stencil(size_t n, double *RHS, double *P) {

  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n)
    return;
  // for (int j = 1; j < jmaxLocal + 1; j++)
  for (int i = 1; i < imax + 1; i++) {

    r = RHS(i, j) - ((P(i - 1, j) - 2.0 * P(i, j) + P(i + 1, j)) * idx2 +
                     (P(i, j - 1) - 2.0 * P(i, j) + P(i, j + 1)) * idy2);

    P(i, j) -= (factor * r);
    res += (r * r);
  }
}

__global__ void reduce_(size_t n, double res) {}