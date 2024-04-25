# Fine-Tuning Self-Supervised Language Models with Latent Output From Tools

## Introduction

Language models have become a cornerstone of modern natural language processing, offering remarkable capabilities in generating and understanding human language and text. A large source of text is in the form of application programming interfaces (API) and these language models have proven to be proficient at generating API calls.

However, they are not typically pre-trained or fine-tuned during self-supervised learning on the output of the API calls, where we can treat both the input and output from the API call as **latent** and is thus only used to **enrich** the context or training data
with additional information that may faciliate next-token prediction.

Since it is trained end-to-end, using the same LLM to both generate the API calls and to predict the next token, it is also hoped that the LLM will learn to generate API calls that are more likely to help it produce the next token. This may be actually quite nuanced. For instance, given a training example:

> Mary struggled with alegebra. When she would try to solve variables
> in terms of, say, $y$, in $3 y = 3 x + 3$, she would often see that
> she needs to divide both sides by $3$ to isolate the $y$ on the LHS,
> but she would predictably only divide the $x$  by $3$, yielding the
> incorrect result $y = x + .

The LLM may have competently learned to generate the API call `solve("y", "3 y = 3 x + 3")`, yielding the **enriched** training data:

> ```latent
> Input: `solve("y", "3 y = 3 x + 3")`
> Output: `y = x + 1`
> ```
> ```observable
> Mary struggled with alegebra. When she would try to solve variables
> in terms of, say, $y$, in $3 y = 3 x + 3$, she would often see that
> she needs to divide both sides by $3$ to isolate the $y$ on the LHS,
> but she would predictably only divide the $x$  by $3$, yielding the
> incorrect result $y = x +
> ``` 

The output from the API call is **latent** in the sense that it is not used to compute the loss, but is ostensibly only used to help the LLM predict the next token, `3`, from the obserable data. However, in this case, the output from the
API call suggests the output should be `1` (which is technically correct, but not in the given co   ntext), and thus may result in reducing prediction quality.

The LLM must be trained end-to-end on latent enrichments from tools so that it learns not only does it learn how to use APIs, but **when** to use them based on the context. The LLM may ignore the latent information from the API call, or use it appropriately, to predict the correct next token, but at face value it seems like the latent information may induce a bias that may be difficult to overcome. Ideally, the LLM would either not use the `solve` to enrich the context with latent data, or it would use
some *other* tool. For instance, it may have used a `let's think step by step` tool to generate the following **enriched** training data:

> ```latent
> Let's think step by step:
> We want to solve the equation $3 y = 3 x + 3$ for $y$.
> To do this, we need to isolate $y$ on the LHS.
> What are the steps we need to take to isolate $y$?
> 1. Divide both sides by $3$.
> 2. This yields $3y / 3 = 3x / 3 + 3 / 3$.
> 3. However, we know that Mary probably only divided the $x$ by $3$,
>    so we need to make sure we only divide the $x$ by $3$.
> 4. This yields $3y / 3 = 3x/3 + 3$.
> 5. When we simplify, we get $y = x + 3$.
> We see that she probably incorrectly solved for $y$ as $y = x + 3$. 
> ```
> ```observable
> Mary struggled with alegebra. When she would try to solve variables
> in terms of, say, $y$, in $3 y = 3 x + 3$, she would often see that
> she needs to divide both sides by $3$ to isolate the $y$ on the LHS,
> but she would predictably only divide the $x$  by $3$, yielding the
> incorrect result $y = x +
> ``` 

In this case, the LLM may have learned to use the `let's think step by step` tool to generate the **enriched** training data, and thus may have learned to predict the correct (but technically wrong) next token, `3`, from the obserable data.

Imagine now, though, that we have the following continuation in the training data:

> Mary struggled with alegebra. When she would try to solve variables
> in terms of, say, $y$, in $3 y = 3 x + 3$, she would often see that
> she needs to divide both sides by $3$ to isolate the $y$ on the LHS,
> but she would predictably only divide the $x$  by $3$, yielding the
> incorrect result $y = x + 3$.
>
> To help her understand her mistake, her teacher, Mr. Smith, pointed
> out that she only divided the $x$ by $3$, and not the $3$ on the RHS.
> He reminded her that the equation is like a scale, and that she needs
> to do the same thing to both sides to keep the scale balanced.
> If she divides the LHS side by 3, she needs to divide the entire RHS
> by 3 as well: LHS /3 = RHS / 3. He says to her to place a parenthesis
> around the RHS, and then divide the expression in the parenthesis by
> 3. This yields LHS / 3 = (RHS) / 3. He works through the problem with
> her, and they get the correct answer, $y = x +

Now, for this particular training data, we might have for the following
**enriched** training data:

> ```latent
> Input: `solve("y", "3 y = 3 x + 3")`
> Output: `y = x + 1`
> ```
> ```observable
> Mary struggled with alegebra. When she would try to solve variables
> in terms of, say, $y$, in $3 y = 3 x + 3$, she would often see that
> she needs to divide both sides by $3$ to isolate the $y$ on the LHS,
> but she would predictably only divide the $x$  by $3$, yielding the
> incorrect result $y = x + 3$.
>
> To help her understand her mistake, her teacher, Mr. Smith, pointed
> out that she only divided the $x$ by $3$, and not the $3$ on the RHS.
> He reminded her that the equation is like a scale, and that she needs
> to do the same thing to both sides to keep the scale balanced.
> If she divides the LHS side by 3, she needs to divide the entire RHS
> by 3 as well: LHS /3 = RHS / 3. He says to her to place a parenthesis
> around the RHS, and then divide the expression in the parenthesis by
> 3. This yields LHS / 3 = (RHS) / 3. He works through the problem with
> her, and they get the correct answer, $y = x +
> ``` 

In this case, the `solve` tool generates information that is not only
correct, but is also useful for predicting the next token, `3`, from
the observable data.

The trick is to train the LLM end-to-end on the **enriched** training
data in an appropriate way, based on the context. This self-supervised
learning approach will hopefully induce the LLM to learn to use the
tools appropriately, and to learn to predict the next token from the
observable data based on this potentially enriched context.

## Latent Code Model

Suppose we language model produces latent Python code.
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
