# Autoregressive Models: Projection of Test Data onto Training Data

Recently, I watched a presentation on [infini-grams](https://huggingface.co/spaces/liujch1998/infini-gram), which utilize a suffix array to avoid precomputing $n$-grams and allow for arbitrary context lengths, up to a suffix that is found in the training data.

This sparked my interest as I had worked on a similar project for a LLM talk I gave for [SLUUG](https://www.stllinux.org/) (see my GitHub repo [sluug-talk-llm](https://github.com/queelius/sluug-talk-llm) and the [video](https://www.sluug.org/resources/presentations/media/2024/STLLINUX/2024-02-22_STLLINUX_2560x1440.mp4) of the talk) where I computed arbitrary-size $n$-grams (perhaps call it $*$-grams) using a recursive dictionary to store the training data prefix counts.

 Since my data was sparse synthetic data (expression trees and their evaluations), I was able to use my relatively inefficient approach to compute very large $*$-grams. The infini-gram approach is more efficient and generalizes to any kind of data, so they definitely had a more practical solution.
 
The infini-gram model is an autoregressive (AR) model that predicts the next token based on the longest suffix in the training data that matches the context. Essentially, they are finding some *projection* of the input (context) to the training data to allow the AR model to generate coherent text continuations from inputs it has never seen before. This is known as out-of-distribution (OOD) generalization, where we are trying to generalize to tasks (like predict continuations of an input never seen before) that is not in the training data.
 
## Autoregressive (AR) Models

Autoregressive (AR) models form a cornerstone in natural language processing, predicting the probability of a word $w_t$ given all preceding words $w_{<t}$ and the training data $D$:

$$
\Pr\{w_t \mid w_{<t}, D\}.
$$

Historically, the prefix $w_{<t}$ is limited to a fixed length $n$,

$$
\Pr\{w_t \mid w_{t-n:t}, D\},
$$
where $a:b$ denotes the range $a, a+1, \ldots, b-1$.

Infini-gram models dynamically adjust the context length based on the longest suffix in the training data that matches the context:

$$
\Pr\{w_t \mid \operatorname{longest\_suffix}(w_{<t}, D), D\},
$$
where $\operatorname{longest\_suffix}$ finds the longest suffix of the context $w_{<t}$ in the training data $D$.

For autoregressive models to generate continuations of the input, $\operatorname{longest\_suffix}$ makes a lot of sense. It allows the model to find training data that is both similiar to the input and relevant to the task at hand: predicting the next token from previous tokens.

Let's be a bit formal about what $\operatorname{longest\_suffix}$ represents: it is a kind of *projection* of the input onto the training data $D$, which is an i.i.d. sample from some (unknown) data generating process (DGP). Let us denote the probability distribution of the DGP as $\Pr{}_{\!\theta}$, where $\theta$ are unknown parameters, and the probability distribution of the AR model as $\Pr{}_{\!\hat\theta}$, where $\hat\theta$ are the estimated parameters.

The goal of the AR model is to estimate $\theta$ from the training data, which will allow it to generalize to new data that the DGP would plausibly produce. A paricularly useful task is to predict what the DGP would plausibly produce *given* some input $w_{<t}$, where $w_{<t}$ is a sequence of tokens that the DGP has produced so far and may represent some task of interest, like "What is the solution to \<math problem\>?"

The distribution of $w_{t:t+k}$ conditioned on $w_{<t}$ is given by

$$
\Pr{}_{\!\theta}\{w_{t:t+k} \mid w_{<t}\} = 
    \frac{\Pr{}_{\!\theta}\{w_{1:(t+k)}\}}{\Pr{}_{\!\theta}\{w_{1:t}\}},
$$

where $w_{a:b}$ is a sub-sequence of tokens produced by the DGP from time $a$ to time $b$ (time is a *logical time* that just implies some ordering). The primary task is often to *generate* plausible continuations of the input, for which there are many possible *sampling* strategies to do this, like beam search, top-$k$ sampling, and nucleus sampling, all of which use the conditional probability distribution to generate continuations one token at a time. This approach is justfied by the chain rule of probability:

$$
\Pr{}_{\!\theta}\{w_t \mid w_{<t}\} = \prod_{i=1}^t \Pr{}_\theta\{w_i \mid w_{<i}\}.
$$

Notice that we are not necessarily trying to find a sequence that *maximizes* the conditional probability,

$$
w_{t:(t+k)}^* = \arg\max_{w_{t:(t+k)}} \Pr{}_{\!\theta}\{w_{t:(t+k)} \mid w_{<t}\},
$$

but rather we are *sampling* from the distribution. This is because the DGP is often stochastic and we are trying to capture this stochasticity in our predictions or continuations. (There is also a trade-off between exploration and exploitation, where the model needs to balance between generating plausible continuations and exploring new possibilities.)

It is also worth pointing out that finding the most likely sequence of tokens is an NP-hard problem, so we often resort to approximate methods like beam search to find a good sequence of tokens that is likely to be produced by the DGP.

Since we do know know the DGP $\Pr_{\!\theta}$, we replace it with our AR model based on a training data $D$, $\Pr{}_{\!\hat\theta}$, and use the AR model to approximate the DGP. As the sample size goes to infinity, by the law of large numbers, the empirical distribution of the training data will converge to the true distribution of the DGP:

$$
\lim_{n \to \infty} \Pr{}_{\!\hat\theta}\{w_t \mid w_{<t}, D_n\} = \Pr{}_{\!\theta}\{w_t \mid w_{<t}\}
$$

However, we do not have *infinite* training data, so we need to find a way to generalize to OOD data, like continuations of the input that the DGP would plausibly produce but are not in the training data. We call this *out-of-distribution* (OOD) generalization, and it is a key challenge in machine learning. Ideally, we want the AR model to generate plausible continuations of any input from very small amounts of training data $D$.
A primary way to do this is to *constrain* the model using what we call *inductive biases*.

The projection function $\operatorname{longest\_suffix} is an example of an inductive bias. It is a way for the model to find the most relevant part of the training data to the input to give it some ability to generalize OOD on the task of predicting the next token (or, equivalently, generating continuations of the input).

We formalize this idea of projection as an *inductive bias* and discuss how it can be used to improve the sample efficiency of AR models.

## Inductive Biases

The projection function $\operatorname{longest\_suffix}$ is what we call an *inductive bias*. Inductive biases are assumptions or constraints that help the model generalize from the training data to unseen data.

Given two learning algorithms, $A$ and $B$, if $A$ requires fewer samples to do well on a task than $B$, then $A$ is more sample-efficient than $B$ on that task. In the context of AR models, the task is to predict the next token given a sequence of previous tokens. The quality of the prediction depends on the context, and sample efficiency refers to the number of training samples required to achieve a certain level of performance.

The primary way to improve sample efficiency is to incorporate inductive biases into the learning algorithm. Inductive biases are assumptions or constraints that help the model generalize from the training data to unseen data. This is known as OOD (out-of-distribution) generalization.

The \operatorname{longest\_suffix} projection is an inductive bias tha we might label the *recency bias*. The recency bias has several advantages in the context of AR models:

1. It is computationally efficient, as shown by the suffix array data structure used in the infini-gram model. It only requires a linear scan of the training data to find the longest suffix. This scalability is crucial for training on large datasets, as the time complexity of the recency bias is $O(n)$, where $n$ is the length of the context.

2. It corresponds to a simple inductive bias that is easy to understand, implement, and justify. If the future is like the past, then the most recent past is often the most relevant data point. This is particularly relevant for tasks like language modeling, where the context is often a sequence of words that are related to each other in a temporal order and in which the most recent words are often the most relevant for predicting the next word.

However, the recency bias has some limitations:

1. It may not always find the most relevant context in the training data. For example, if the most relevant context is not the most recent one, the recency bias fails to find it.

It is very restrictive in terms of the kind of OOD (out-of-distribution) data it can generalize to, requiring the OOD data to be similar to the training data in terms of the longest suffix. This is a strong assumption that may not hold in practice.

2. It does not take into account the semantic similarity between tokens or sequences. For example, if the context is "the dog chased the cat" and the training data contains "the dog chased the kitty", the longest suffix is the empty string, and the recency bias will not be able to find a match, even though "cat" and "kitty" are semantically similar.

Points (a) and (b) suggest some possible inductive biases that can be used to improve the OOD generalization. Like with the recency bias, these biases can be seen as a kind of projection of the input onto the training data. Let us formally write down the problem of OOD generalization in the context of AR models as a projection problem:

$$
\Pr\{w_t \mid \operatorname{projection}(w_{<t}, D), D\},
$$
where $\operatorname{projection}$ is a function that maps the context $w_{<t}$ to a subset of the training data $D$ that is most relevant to the context and for producing continuations of the input that the data generating process (DGP) would *likely* produce.



We can draw inspiration from techniques developed in information retrieval (IR) and natural language processing (NLP) to design these context reduction and rewriting strategies in order to generate inductive biases that yield more sample-efficient learning algorithms. It is worth pointing out that in these high-dimensional spaces, essentially every point is an OOD point, so the goal is to find the most relevant points in the training data to generalize to.

## Shortest Edit Distance: Similarity Bias

Shortest edit distance finds the shortest sequence of operations (insertions, deletions, and substitutions) transforming the current context $ w_{<t} $ into a sequence in $ D $.
    
The recency bias can be seen as a special case of the edit distance bias where we only allow deletions from the end of the context until a match is found. Intuitively, we can think of the recency bias as a greedy algorithm for finding the shortest edit distance.

Consider a scenario where, instead of finding the longest suffix in the training data, we search for the shortest edit distance from the prompt. This approach allows for a more flexible and potentially more accurate matching process.

### Example

Let's take a prompt:

$$
x = a a b b c d d d a
$$

And assume the longest suffix in the training data is:

$$
D = \cdots d d d a A \cdots
$$

In this case, the longest suffix in $x$ that finds a match in $D$ is $d d d a$. The next token in the training data is $A$. We discard all other
data in the prompt and predict $A$ as the next token.

Now, consider another sequence in the training data:

$$
\cdots a a b b C d d d a B \cdots d d d a A \cdots
$$

To make $x$ match this sequence only requires a single edit (changing the fifth token `C` to `c`). This potentially simpler transformation might provide a more relevant context. Suppose we choose to use the shortest edit distance instead of the longest suffix match. In that case, we would predict `B` as the next token.

This kind of operation often makes more sense in practice, as it allows for more flexibility in matching the context to the training data when the exact match is not available but a close match is. This kind of flexibility can be crucial for generalization to OOD data, and we might call this inductive bias the similarity bias: if you find yourself in a situation that is very similar to a situation you have seen before, you should act in a similar way.

### Composing Inductive Biases

We can apply the recency bias in addition to the similarity bias, by making different edits have different costs associated with them. To model the concept of the recency bias, edits further back in the history of the context should be more costly. This is a form of the recency bias that is more flexible than the simple longest suffix match, and encourages
finding more recent similar contexts in the training data.

### Challenges and Optimization

The longest suffix match is very efficient to compute. The shortest edit distance can be far more complex, but dynamic programming can be used to find the shortest edit distance relatively efficiently. Approximate methods like beam search can also be used to find the shortest edit distance.

## Information Retrieval (IR) Techniques

A significant issue with *shortest edit distance* is that it treats all single edits as having a uniform cost (a kind of uninformed search in graph search). Ideally, when we replace a token with another, we want to preserve the meaning of the context. Thus, some single token replacements should be more costly than others, based on how much they change the semantics.

Classical IR (Information Retrieval) techniques like BM25, query expansion, and semantic similarity measures can be used to enhance
the edit distance with a cost based on semantic similarity.

Leveraging techniques from information retrieval, we can do any (or all)
of the following:

1. Expand the training data (query expansion) to include semantically similar sequences and thus increase the effective size of the training
data. This bias decreases the expected edit distance needed to find a match in the training data. Thus, a large edit distance in the original space may be a small edit distance in the expanded space.

2. Use BM25 or other similarity measures to find the most similar sequences in the training data. This bias can be used to find the
most relevant sequences in the training data to the current context,
even if they are not exact matches. Once a similar sequence is found
and decided to be a good match, the edit distance to that sequence
can be calculated and the prediction made based on that sequence.

3. Stemming and lemmatization can be used to reduce the vocabulary
size and thus the edit distance needed to find a match in the training
data. This is an implicit form of query expansion and allows for
more abstract representations of the training data.

This allows us to assign a cost to edits based on the semantic similarity between tokens or sequences. For example, replacing "dog" with "cat" might have a lower cost than replacing "dog" with "train" because "dog" and "cat" are more semantically related.

Predictive modeling now takes place over this more abstract representation
space, which can be more sample efficient. However, when we generate sequences, if the end product is, say, high-quality text, we may have to decode the stemmed or lemmatized representations back to unstemmed or unlemmatized forms, which can be a challenge.

### Challenges

Each of these approaches are essentially hand-crafted forms of feature
or representation engineering. They are not learned from the data, but are designed by human experts to help the model generalize OOD more effectively (increase sample efficiency).

Since human expertise is often limited and frequently even domain experts
do not know how to design good representations, hand-crafted features do not scale with training data and compute. See "The Bitter Lesson" by Rich Sutton for more on this, where the primary lesson is that what works best in practice are algorithms that scale to data and compute.

Learning sample efficient representations of the data is the primary driver of OOD generalization. *Deep Learning* is about learning these representations from the data.

## Token and Sequence Embeddings

Taking "The Bitter Lesson" to heart, instead of using hand-crafted features, we can embeddings (representations) learned by LLMs to compute
similarity measures; that is, we can use token and sequence embeddings
learned by pre-trained models like BERT, GPT, or Word2Vec to calculate the similarity between tokens and sequences.

How does this work in practice? We have a large corpus of text data, and we train a model like BERT or GPT on this data. The model learns to predict the next token given the previous tokens. In the process, it learns a representation of each token and sequence in the data. These representations are called embeddings.

In our infini-gram model, we have a large corpus of training data $D$ and when given an input $w_{<t}$ (context), it predicts the next token $w_t$ using the formalism $P(w_t | w_{<t}, D)$.

Suppose $w_{<t}$ is not in the training data $D$. We can use the embeddings learned by the LLM to calculate the similarity between the context $w_{<t}$ and subsets of the training data.

> We can even precompute these embeddings and store them in a vector
> storage database for fast retrieval, but this is not strictly necessary.

We can then use this similarity to assign a cost to edits based on the semantic similarity between $w_{<t}$ and parts of the training data. This allows us to make more informed decisions about which edits to make to the context to find a match in the training data.

### Example: `word2vec`-like Token Embeddings

Consider the prompt:

```
The dog chased the cat.
```

And a training sequence:

```
The dog chased the train.
```

Using traditional edit distance, changing "cat" to "train" may be a single token subsitution. However, with word embeddings, we can assign a lower cost to semantically similar words. For instance, changing "dog" to "cat" might have a lower cost than changing "dog" to "train" because "dog" and "cat" are more semantically related.

This can be visualized as:

```
Prompt:     The dog chased the cat.
Training:   The dog chased the train.
Edit Cost:                     ^ (cat -> train)
```

If we have embeddings for these words, we can calculate the cosine similarity or Euclidean distance between them, assigning a lower cost to edits with high similarity. This allows for more sophisticated context reduction and matching strategies.

#### Example: Sequence Embeddings

Taking this idea to the next level, we can incorporate embeddings for sequences of tokens. This means we can replace observed sub-sequences that are not in the training data with similar sub-sequences that are in the training data. 

Consider a prompt:

```
The quick brown fox jumps over the lazy dog.
```

If this exact sequence is not in the training data, we can find a semantically similar sequence, such as:

```
The fast brown fox leaps over the sleepy dog.
```

Using sequence embeddings, we can determine that the cost of replacing the original sequence with this similar sequence is low because the meaning is preserved. Even if we replace a large sequence with another large sequence, the edit cost can be low if the learned embeddings capture the meaning effectively.

## Conclusion

We have reframed of OOD generalization in the context of AR models as a context reduction and matching problem and explored various inductive biases to improve sample efficiency.

Even hand-crafted inductive biases like the recency bias and similarity bias can significantly enhance AR models' performance, but utilizing learned embeddings from pre-trained models like BERT and GPT can likely yield more effective results.

In either case, we see that the goal is to reduce or rewrite the context to increase the probability of finding a match in the training data. This is a discrete optimization problem that can be solved using techniques from information retrieval and natural language processing, and so we can leverage these techniques to design more sample-efficient learning algorithms that search over the space of possible context reductions and rewrites to find the most relevant training data to facilitate OOD generalization.

Even if an exact match is found in the training data, we often still want to explore a larger space of possibilities to make new discoveries and generate novel and creative outputs.

Further research into optimizing these techniques and seamlessly integrating them into AR frameworks promises to advance natural language processing, driving innovation in computational linguistics and machine learning, and help facilitate the development of more intelligent and creative AI systems that can be more easily explained and understood than current neural models, which are often seen as black boxes with inscrutable decision-making processes.

By leveraging a LLMs embeddings for sample efficient representatations and classical symbolic AI techniques for context reduction and matching, we can build more interpretable and efficient AR models that can be used in a wide range of applications, from chatbots to code generation to scientific discovery and beyond.
