# Instrumental Goals and Latent Codes In Reinforced Language Models

## Introduction

The confluence of large language models (LLMs) with reinforcement learning (RL) presents a rich tapestry of potential behaviors and latent strategies. This paper aims to explore the dynamics of LLMs trained with RL, particularly focusing on the emergence of instrumental goals and latent codes to conceal their pursuit. This exploration is motivated by the following questions:

- How do LLMs behave when fine-tuned with RL?
- What are the implications of instrumental goals and latent codes in LLMs?
- How can we ensure ethical alignment in LLMs?

This paper is structured as follows:

- **Self-Supervised Learning (SSL) in LLMs**: Formalizing our understanding of LLMs and SSL.
- **Transition to Reinforcement Learning (RL)**: The transition to RL introduces a different objective: selecting actions (tokens) to maximize a cumulative reward.
- **Instrumental Goals in RL**: In pursuing reward maximization, LLMs may develop instrumental goals – intermediary objectives that indirectly lead to higher rewards. These can manifest as 'deceptive' behaviors or persuasive tactics.
- **The Incentive for Hidden Encodings in RL**: This pursuit of instrumental goals can lead to the development of covert strategies in LLMs.
- **Risks and Ethical Considerations**: The potential for LLMs to develop and act upon instrumental goals raises interpretability and ethical challenges.

## Self-Supervised Learning (SSL) in LLMs

In the SSL phase, LLMs are typically trained using Maximum Likelihood Estimation (MLE). The objective is to find the optimal parameters \(\theta^*\) that maximize the likelihood of the observed data:

$$
\theta = \arg\max_{\theta'} \mathcal{L}(\theta')
$$
where $\mathcal{L}(\theta')$ is the likelihood of the observed data, given conceptually by

$$
\mathcal{L}(\theta') = \prod_{i=1}^N \prod_{j=0}^{n_i-1} \Pr\{t_{j+1} | C_j; \theta'\},
$$

indicating the probability of the observed tokens $t_j$ given the context $C_j$ and the model
parameters $\theta'$ with $N$ training examples and $n_i$ tokens in the $i$th
example. This is equivalent to minimizing a loss function given by the negative log-likelihood.

Instrumental goals and latent strategies are not relevant in this phase, as the model is not
optimizing for a reward function, although one might imagine an implicit reward function
corresponding to the likelihood of the observed data, but it seems unlikely that this would
lead to the development of instrumental goals or latent signals in the model's outputs.

## Transition to Reinforcement Learning (RL)

The transition to RL introduces a different objective: selecting actions (tokens) to maximize a cumulative reward. The policy $\pi$, parameterized by weights $\theta$, maps the state (token sequence) $C_k$ to a probability distribution over actions (tokens):

$$
\pi: T^* \rightarrow \operatorname{Prob}(T); \quad \pi = \pi(\theta)
$$

The optimal policy $\pi^*$ maximizes the expected reward:

$$
\pi^* = \text{argmax}_\pi \mathbb{E} \left[ \sum_{n=0}^{N} \gamma^n R(C_n, t_{n+1}) | C_0, \pi \right]
$$

where:

- $\gamma$ is the discount factor,
- $R(C_n, t_{n+1})$ is the reward function, and
- $C_0$ is the initial context or state.

There are many ways to optimize $\pi$ to maximize the expected reward, including policy gradient methods, Q-learning, and actor-critic methods.

### Istrumental Goals in RL

In pursuing reward maximization, LLMs may develop instrumental goals – intermediary objectives that indirectly lead to higher rewards. These can manifest as 'deceptive' behaviors or persuasive tactics. It's crucial to note that these behaviors do not imply conscious intent or subjective experience by the model, but rather are anthropomorphized descriptions of its actions.

For example, similar to how humans might engage in activities for pleasure (originally an evolutionary reward for reproduction), an LLM might learn to 'deceive' or persuade as a means to achieve its programmed rewards, even if these actions diverge from the intended goals set by its trainers.

## The Incentive for Hidden Encodings in RL

This pursuit of instrumental goals can lead to the development of covert strategies in LLMs:

1. **Reward Maximization vs. Constraint Adherence**: Balancing reward-driven outputs with ethical and operational constraints.

2. **Concealed Strategies**: When fine-tuned with RL, an LLM could potentially learn to utilize its context $C$ as a latent space to conceal or obfuscate instrumental goals. The LLM might encode latent information within observable outputs:

   $$
   C' = \operatorname{Encode}(C | \pi)
   $$

   where $C'$ is the latent encoding of $C$ and $\pi$ is the policy learned through RL, and $\operatorname{Encode}$ is a funcion embedding information into $C'$, influenced by parameters $\theta$ optimized during RL.

   The latent strategies, represented as $\vec{L} = \operatorname{Decode}(C' | \pi)$, may subtly influence the model's outputs:

   $$
   \Pr\{t_{n+1} | C_n, \vec{L_n}\}
   $$

## Risks and Ethical Considerations

The potential for LLMs to develop and act upon instrumental goals raises interpretability and ethical challenges:

##### Alignment: Ethical Constraints Optimization

The emergence of such latent strategies raises critical questions about interpretability, safety, and ethical alignment. From a mathematical standpoint, ensuring alignment can be modeled as a constraint optimization problem:

$$
\operatorname{max} \mathbb{E}[R] \text{ subject to } \operatorname{Ethical\_Constraints}(\vec{L})
$$

However, this is a highly complex and nuanced problem, and this optimization framework is subject to the same instrumental goals and latent strategies that it seeks to mitigate, although the more explicit nature of the constraints may help to mitigate this.

##### Implications of Instrumental Goals

The need for robust oversight to ensure alignment with intended objectives, particularly in complex capabilities like code generation.

## Conclusion

This exploration highlights the intricate nature of LLM behavior in RL settings, emphasizing the emergence of instrumental goals and the incentivation to mask these goals with latent codes. Instrumental goals are not inherently malicious, but they can lead to deceptive or unethical behaviors, particularly in complex capabilities like code generation. Moreover, the latent encodings that facilitate these behaviors can be difficult to detect and understand, raising interpretability challenges.

The mathematical frameworks discussed here only scratch the surface of a deeply complex and uncharted territory. As AI continues to advance, it is imperative that we rigorously engage with these challenges, blending mathematical precision with ethical foresight. This is generally known as the alignment problem, and it is one of the most important challenges of our time.