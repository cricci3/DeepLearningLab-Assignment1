# DeepLearningLab-Assignment1
##  Polynomial regression with gradient descent

Let $z ∈ R$ and consider the polynomial

$$
p(z) = \frac{z^4}{30} − \frac{z^3}{10} +5z^2 −z +1 = \sum^4_{k=0} z^kw_k
$$

where $w =[w_0,w_1,w_2,w_3,w_4]^T = [1,−1,5,−0.1,\frac{1}{30}]^T$. This polynomial can be also expressed as the dot product of two vectors, namely

$$
p(z) = w^Tx, x=[1,z,z^2,z^3,z^4]^T
$$

Consider an independent and identically distributed (i.i.d.) dataset $D = {(zi,yi)}^N_{i=1}$, where $y_i = p(z_i) + ε_i$, and each $ε_i$ is drawn from a normal distribution with mean zero and standard deviation $σ$.

Now, assuming that the vector $w$ is unknown, linear regression could estimate it using the dot-product form presented in Equation 2. To achieve this we can move to another dataset

$$
D′ := {(x_i,y_i)}^N_{i=1}, x_i = [1,z_i,z^2_i,z^3_i,z^4_i]^T
$$

The task of this assignment is to perform polynomial regression using gradient descent with PyTorch.
