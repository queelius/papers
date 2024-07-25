
# softAR

## Overview
This repository is dedicated to the development and experimentation of a novel autoregressive (AR) model for language generation. The model aims to balance nuanced expression with coherent, grounded language generation by utilizing a mix of aggregated embeddings, hard tokens, and soft tokens in its context management strategy.

In an AR language model, tokens are generated using the following scheme:

$$
\Pr(X_{n+1} = x_{n+1} | x_n, \ldots, x_{n-k+1})
$$

This is $k$-th order Markov chain, where the history of states are given by the last $k$ tokens.

We consider here the idea of constructing an AR model over *soft* tokens, or more precisely, the embedding space.
Supupose the embedding space is $R^m$, then we have an AR model over real vectors in $R^m$.

### Latent Embedding Space

As a first experiment, we can consider creating an AR model just over these embeddings, but train it on hard token data.
In this case, at inference time, every sequence of vectors is OOD (out-of-distribution), since we're dealing with real vectors rather than symbols. At some point, we'll want to convert this sequene of "soft" (sub-symbolic) tokens to hard (symbolic) tokens, so that the result can be communicated to others. We might call this "collapsing the wave function" since, prior to the collapse, the semantics of the sequence are esedntially "non-utterable".

This is similar to the problems humans face. They often have trouble finding the words to express their thoughts and feelings. We face the same problem with this soft-token AR model. It "thinks" in this latent embedding space, but to communicate its results, iot must map this latent information to observable symbolic data that is easier to communicate, as the symbols are grounded to external things that we can point to. It's discrete, and so easier to "localize" in an otherwise latent space of ideas.

#### Collapse of the Wave Function
The collapse of the wave function can take on many forms. We could use a greedy algorithm, where for each latent "soft" token, we map it to the nearest grounded symbol (token). This will almost surely lose a lot of nuance and information.

Another approach is to use a layer in the neural network in which all of the context has been mixed together, and map that $R^{q \times p}$ object to a point in the embedding space $R^m$. We actually have the training data for this: when we train on hard token data to predict the next hard token, we can map the objects in this layer to the embedding of the hard token. Clearly, we lose a lot of information in this process, and the training data itself is only over the symbolic data, but when we later fine-tune the system, say using RL, we can allow the model to generate more nuanced reasoning steps. Also, even if it is only trained on hard token data, it may still genearlize out of distribution at inference time to capture more nuanced patterns for better predictions when we "collapse" the token sequences.

### Coherence

At inference time, we may want to "ground" the tokens not just for communication to others, but for itself too. The more out of distribution (OOD) a sequence is from the training data, the less reliable and coherent the sequences will be. So, instead, we can selectively collapse soft tokens in the context to make the sequences more like the training data sequences.

### Compression

Since our AR model is over real vectors in the embedding space $R^m$, there is an opportunity to take advantage of the rich semantic structure in this space to do meaningful operations that *compress* the representation of the context.
For instance, we can add two embeddings together to compress the representation. For instance, it is well known that these embeddings learn a sort of linear algebra over the embeddings, such that $\text{king} - \text{man} \approx \text{queen} - \text{female}$. We can apply these kind of opeerations to embeddings in the context to compress the representation, or even retroactively enrich it.

## Pre-Training

During the pre-training stage, we have two options. We could forget about the generation of soft-tokens entirely, and only generate hard tokens, and later on when fine-tuning, we could still take advantage of the insight that we can use "soft" context (take the hard token embeddings and add them together, for instance, to avoid throwing away old context entirely). This is interesting and possibly very effective all by itself.

In this experiment, though, we are going to assume that, ultimately, at inference time, we'll be taking advantage of the potentially extra information in "soft" tokens. In this case, we have to structure the loss function differently. The output will now be a single vector, $y \in R^m$, and we can either train it using MSE (mean squared error) against hard token pre-training data

## Context Management Strategy
We collect these insights here to propose a new way of representing the context, such that hopefully the context is both more subtle and allows for more nuance and compression.

The model's context window is structured as follows to optimize the balance between nuance and coherence:

1. **Aggregated Embeddings as Compressed History**:
   - The last position in the context window holds an aggregation of all previous token embeddings.
   - This provides a highly compressed representation of the entire context history.

2. **Segmented Context Summations**:
   - Positions 2 second and third positions in the context window are dedicated to separate summations of different segments of the context history.
   - These positions offer a less compressed view, capturing more recent or relevant context nuances.

3. **Hard Tokens for Coherence and Grounding**:
   - Positions four through one hundred consist of hard tokens, grounding the model's output in more concrete, symbolic language.
   - This part of the context window helps maintain coherence and alignment with the training data's structure and patterns.

4. **Soft Tokens for Nuanced Thinking**:
   - The remaining positions are reserved for soft tokens, allowing for more nuanced and flexible language generation.
   - Soft tokens can capture subtle variations in meaning, offering richer expression.

## Objectives
- To explore the balance between nuanced expression and coherent language generation.
- To experiment with different representations within the context window to enhance the model's performance.
- To evaluate the effectiveness of aggregated embeddings in retaining essential context information.

## Experimentation
Initial experiments will be conducted using NanoGPT or similar frameworks to test the feasibility and effectiveness of the proposed context management strategy.

## Contributing
Contributions are welcome! Whether it's feature suggestions, code contributions, or discussions about the theoretical aspects of the model, feel free to participate.

## License
[MIT License](LICENSE)

---

*This README is part of an ongoing experimental project and is subject to change as the project evolves.*
