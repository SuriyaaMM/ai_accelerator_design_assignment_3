### Suriyaa MM (EE23B054)

---

## 1. Problem Statement & Objective
In AI hardware accelerators, multidimensional tensors (such as images and feature maps) are often serialized into flattened representations to be streamed into processing elements like systolic arrays. The objective of this assignment is to manually implement and verify the address indexing logic required to flatten a multi-dimensional tensor $I \in \mathbb{R}^{B \times C \times H \times W}$ into a 2D tensor of size $B \times (CHW)$, and then successfully reconstruct it back to its original layout, explicitly computing the element-wise reconstruction error to verify correctness.

## 2. Algorithms and Pseudocode
To convert between layouts without relying on built-in library functions, explicit strided addressing logic was used.

### 2.1 Flattening Algorithm: $B \times C \times H \times W \longrightarrow B \times (CHW)$
To map a 4D element index $(b, c, h, w)$ to a 2D index $(b, k)$ in the flattened tensor, we compute the linear 1D offset $k$ within a specific batch $b$. The channel stride is $H \times W$ and the height stride is $W$.

**Pseudocode:**
```text
function FLATTEN(I_4D):
    B, C, H, W = shape(I_4D)
    CHW = C * H * W
    I_2D = allocate_zeros(B, CHW)
    
    for b = 0 to B-1:
        for c = 0 to C-1:
            for h = 0 to H-1:
                for w = 0 to W-1:
                    k = c * (H * W) + h * W + w
                    I_2D[b, k] = I_4D[b, c, h, w]
                    
    return I_2D

```

### 2.2 Reconstruction Algorithm: $B \times (CHW) \longrightarrow B \times C \times H \times W$

Given a flattened offset $k$, we extract the original multi-dimensional indices $(c, h, w)$ using integer division (quotient) and modulo (remainder) operations based on the dimensional strides.

**Pseudocode:**

```text
function RECONSTRUCT(I_2D, B, C, H, W):
    CHW = C * H * W
    I_4D = allocate_zeros(B, C, H, W)
    stride_c = H * W
    stride_h = W
    
    for b = 0 to B-1:
        for k = 0 to CHW-1:
            c = k // stride_c
            rem = k % stride_c
            h = rem // stride_h
            w = rem % stride_h
            
            I_4D[b, c, h, w] = I_2D[b, k]
            
    return I_4D

```

## 3. Manual Indexing Illustration

Consider a small tensor with dimensions **$B = 1, C = 2, H = 2, W = 3$**.
The flattened size for the batch is $CHW = 2 \times 2 \times 3 = 12$.
The strides are:

* Channel stride ($S_c$) = $H \times W = 2 \times 3 = 6$
* Height stride ($S_h$) = $W = 3$

Let's manually trace the mapping for a few elements:

**Example A: Forward Mapping of element at $(b=0, c=1, h=0, w=1)$**

* $k = c \cdot (H \cdot W) + h \cdot W + w$
* $k = 1 \cdot (6) + 0 \cdot (3) + 1 = 6 + 0 + 1 = 7$
* The element maps to `I_2D[0, 7]`.

**Example B: Forward Mapping of element at $(b=0, c=0, h=1, w=2)$**

* $k = 0 \cdot (6) + 1 \cdot (3) + 2 = 5$
* The element maps to `I_2D[0, 5]`.

**Example C: Reverse Mapping of flattened index $(b=0, k=9)$**

* $c = \lfloor 9 / 6 \rfloor = 1$
* $rem = 9 \pmod 6 = 3$
* $h = \lfloor 3 / 3 \rfloor = 1$
* $w = 3 \pmod 3 = 0$
* The element maps back to `I_4D[0, 1, 1, 0]`.

## 4. Experimental Results

Experiments were conducted using samples from realistic datasets (MNIST and CIFAR-10) as well as synthetic Gaussian random tensors to emulate intermediate deep neural network feature maps with high channel counts ($C \in \{16, 64, 128, 500\}$).

The errors were calculated as follows:

* **Max Absolute Error ($E_{max}$)**: $\max \vert{} I[b,c,h,w] - \hat{I}[b,c,h,w] \vert{}$
* **Mean Absolute Error (MAE)**: $\frac{1}{BCHW} \sum \vert{} I[b,c,h,w] - \hat{I}[b,c,h,w] \vert{}$

**Table 1: Experimental results showing the MAE for different classes of image tensors**

| Dataset/Input | B | C | H | W | $E_{max}$ | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| MNIST / sample | 1 | 1 | 28 | 28 | 0.0e+00 | 0.0e+00 |
| CIFAR / sample | 1 | 3 | 32 | 32 | 0.0e+00 | 0.0e+00 |
| Synthetic fmap1 | 1 | 16 | 32 | 32 | 0.0e+00 | 0.0e+00 |
| Synthetic fmap2 | 1 | 64 | 32 | 32 | 0.0e+00 | 0.0e+00 |
| Synthetic fmap3 | 1 | 128 | 32 | 32 | 0.0e+00 | 0.0e+00 |
| Synthetic fmap4 | 1 | 500 | 32 | 32 | 0.0e+00 | 0.0e+00 |
