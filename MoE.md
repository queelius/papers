


# Mixture-of-Experts

## Latent Code Model

In this expert model, the language model produces latent Python code.
This code is then evaluated by a Python interpreter, which produces the observable output. It is this observable output that is used to compute the loss during the fine-tuning phase of a large language
model (LLM).

We are relying on a pre-training phase where it has already developed competency at generating Python code given the right context. When the context is something like "What is the average value of `[3, 0, 1, 300, 2, 94, 93, -1, -2]`?", it will generate code to compute the mean of a list of numbers. The code will be something like:

```python
def mean(x):
    return sum(x) / len(x)
mean([3, 0, 1, 300, 2, 94, 93, -1, -2])
```

We will have a regularizer that will encourage the latent code to be as small as possible while still producing correct output. By Occam's razor, small codes tend to generalize better, as most DGPs are relatively simple. A priori, this could be reasoned about in the following way: suppose the
DGP of reality is given by randomly generating a prefix-free program, and then running it and observing its output, rince and repeat. Then, there are an infinite number of programs that could generate the same output, but the shortest program has a higher probability of being randomly selected (fewer bits) and is thus more likely to be the DGP than a longer program, both of which are compatible with the observed sequence so far. This is the intuition behind the Minimum Description Length (MDL) principle.


What if the output is incorrect? This is an interesting and very important question. On the one hand, it could always produce the correct output by
simply generating code like:

```python
print("The answer is 42")
```



We are then fine-tuning the model to produce code that will predict the next token -- not the next code token, but the next token in the language model. The code is then evaluated by the code evaluator, and the loss is computed based on the output of the evaluator.

We must propagate the errors in the final output back to the latent code, since it is the code itself that generated the final output.

When the final output is wrong, then the code that produced it is wrong. As a theoretical brute-force approach, we could slightly randomize the language model weights (that produced the code in the first place) and then generate a new code. This new code would then be evaluated, and if it produces a better output, then we would use it to update the language model weights. This is a brute-force approach, but we want to use backpropogation to update the language model weights.

L(x) = loss of the final output
L(c) = loss of the latent code
