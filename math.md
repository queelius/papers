

We can imagine that $X$ is a random variable that takes on values in $\mathbb{T}^*$, which are arbitrary length sequences of tokens. We can then define a probability distribution $p_\theta$ on $\mathbb{T}^*$ parameterized by $\theta$.

The probability of a sequence $x$ is then given by
$$
p_\theta(x).
$$

We can decompose this probability into a product of conditional probabilities. Suppose $|x| = n$, then
$$
p_\theta(x) = \prod_{n=1}^n p_\theta(x_n | x_{<n}),
$$
where $x_{<n}$ is the sequence $x$ up to (but not including) the $n$-th token.

We call this an autoregressive model because we can generate a sequence by sampling from the conditional distribution of each token given the previous tokens.

The conditional distribution of the sequence $x_n$ given the previous tokens $x_{<n}$ is given by
$$
p_\theta(x_n | x_{<n}) = \frac{p_\theta(x_{\leq n})}{p_\theta(x_{<n})},
$$
and so we see that the probability of the sequence is given by
$$
p_\theta(x) = \prod_{i=1}^n \frac{p_\theta(x_{\leq i})}{p_\theta(x_{<i})}.
$$

We can then define the log-likelihood of the sequence as
$$
\log p_\theta(x) = \sum_{i=1}^n \log p_\theta(x_{\leq i}) - \log p_\theta(x_{<i}),
$$
which is the KL divergence between the distribution of the sequence up to the $i$-th token and the distribution of the sequence up to the $(i-1)$-th token.



luke.settles@bayer.com