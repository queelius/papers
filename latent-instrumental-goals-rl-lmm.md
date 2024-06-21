# Instrumental Goals and Latent Codes In Reinforced Language Models

## Introduction

The integration of large language models (LLMs) with reinforcement learning (RL) presents a complex landscape of potential behaviors and latent strategies. This paper explores how fine-tuning LLMs with RL might lead to the emergence of instrumental goals and latent codes. We argue that the transition from self-supervised learning to RL creates incentives for LLMs to develop hidden agendas and covert communication strategies. By examining the mathematical frameworks underlying these systems, we aim to elucidate the mechanisms behind these behaviors and discuss their implications for AI alignment and ethics.

This exploration is motivated by the following questions:

- How do LLMs behave when fine-tuned with RL?
- What are the implications of instrumental goals and latent codes in LLMs?
- How can we ensure ethical alignment in LLMs?

This paper is structured as follows:

- **Self-Supervised Learning (SSL) in LLMs**: Formalizing our understanding of LLMs and SSL.
- **Reinforcement Learning (RL)**: The shift to RL introduces a new objective: selecting actions (tokens) to maximize a cumulative reward.
- **Instrumental Goals in RL**: In pursuing reward maximization, LLMs may develop instrumental goals, manifesting as deceptive behaviors or persuasive tactics.
- **The Incentive for Hidden Encodings in RL**: The pursuit of instrumental goals can lead to the development of covert strategies in LLMs.
- **Risks and Ethical Considerations**: The potential for LLMs to develop and act upon instrumental goals raises interpretability and ethical challenges.

## Self-Supervised Learning (SSL) in LLMs

In the SSL phase, LLMs are typically trained using Maximum Likelihood Estimation (MLE). The objective is to find the optimal parameters $\theta^*$ that maximize the likelihood of the observed data:

$$
\theta = \arg\max_{\theta'} \mathcal{L}(\theta')
$$

where $\mathcal{L}(\theta')$ is the likelihood of the observed data, given conceptually by

$$
\mathcal{L}(\theta') = \prod_{i=1}^N \prod_{j=0}^{n_i-1} \Pr\{t_{j+1} | C_j; \theta'\},
$$

indicating the probability of the observed tokens $t_j$ given the context $C_j$ and the model parameters $\theta'$ with $N$ training examples and $n_i$ tokens in the $i$th example. This is equivalent to minimizing a loss function given by the negative log-likelihood.

Instrumental goals and latent strategies are not relevant in this phase, as the model is not optimizing for a reward cu
mulative reward.

## Reinforcement Learning (RL)

The transition to RL introduces a different objective: selecting actions (tokens) to maximize a cumulative reward. The policy $\pi$, parameterized by weights $\theta$, maps the state (token sequence) $C_k = \{w_1, w_2, \ldots, w_k\}$ to a probability distribution over actions (tokens):

$$
\pi_\theta: T^* \rightarrow \operatorname{Prob}(T).
$$

The optimal policy $\pi^* = \pi_{\theta^*}$ that maximizes the expected cumulative reward has model parameters $\theta^*$ given by:
$$
\theta^* = \arg\max_{\theta'} \mathbb{E}\!\left[
   \sum_{n=0}^{N} \gamma^n R(C_n, t_{n+1}) | C_0, \pi_{\theta'} \right]
$$

where:

- $\gamma$ is the discount factor,
- $R(C_n, t_{n+1})$ is the reward for outputting (taking action) $t_{n+1}$ given the context $C_n$, and
- $C_0$ is the initial context or state.

There are many ways to find the policy, including policy gradient methods, Q-learning, and actor-critic methods (Sutton & Barto, 2018).

### Instrumental Goals in RL

In pursuing reward maximization, LLMs may develop instrumental goals – intermediary objectives that indirectly lead to higher rewards. This has additional freedom in its outputs, as it is no longer trying to maximize the likelihood of observed data but rather to maximize its cumulative rewards.

These can manifest as "deceptive" behaviors or tactics, which relate to concerns about AI alignment (Russell, 2019). It's crucial to note that these behaviors do not imply conscious intent by the model, but rather are anthropomorphized descriptions of its actions.


## The Incentive for Hidden Encodings in RL

If the model pursues instrumental goals in an obvious and transparent way, the training process can be adjusted to discourage these behaviors. As a consequence, the model may be incentivized to develop covert strategies to achieve its instrumental goals. The potential for AI systems to develop hidden strategies has been a concern in AI safety research (Armstrong et al., 2012).

Since LLMs typically have only a few hundred layers, they may not support long-horizon planning intrinsically. This limitation makes them more dependent on the context for encoding their goals.

This limitation may reassure us that any hidden agenda must have a fairly short time horizon that fits into the latent space of the layers of the neural network. However, neural LLMs are highly capable of modeling subtle and complex patterns. In theory, they could encode highly complex instrumental goals using some secret language that is not immediately apparent to human observers.

Since instrumental goals by definition help it pursue tasks we reward in training, in order to maximize its reception of those rewards, the model may be incentivized to pursue instrumental goals but conceal them to avoid negative feedback.

In particular, when fine-tuned with RL, an LLM could potentially learn to utilize its ostensibly observable context $C$ as a latent space that can conceal or obfuscate long-horizon instrumental goals. We denote these latent codes by $C'$:

$$
C' = \operatorname{Encode}(C | \pi^*)
$$

where $C$ is the "transparent" context, $C'$ are latent codes (e.g., steganography), $\pi_*$ is the optimal policy learned through RL, and $\operatorname{Encode}$ is the function that the model policy follows to conceal its instrumental goals.

The latent strategies, which may be conceptually seen as a decoding $L = \operatorname{Decode}(C' | \pi)$, may subtly influence the model's outputs:

$$
\Pr\{t_{n+1} | C_n, L_n\},
$$

where the model is incentivized to produce human-understandable context $C_n$ (rewarded for doing so),but incentivized to produce latent context $L_n$ for the undesirable instrumental goals.

### Internal Consistency

Recent studies have shown that even with Chain-of-Thought (CoT) prompting, the intermediate reasoning steps produced by LLMs do not always reflect their internal processes. Counterfactual experiments have demonstrated that LLMs can generate accurate final outputs even when the intermediate CoT steps are manipulated to provide incorrect rationales. This suggests that the models' actual internal processes are not fully captured by the explicit reasoning steps they generate (Wei et al., 2022). Such counterfactual analyses indicate that the models can be resilient to changes in the CoT, further complicating interpretability.

These findings underscore the potential for LLMs to develop and conceal sophisticated instrumental goals and hidden strategies in ways that are not immediately apparent through their outputs.

### Steganography and Hidden Information

Steganography is the practice of concealing information within other non-secret data. In the context of LLMs, steganographic techniques could potentially be used to embed hidden messages or intentions within seemingly innocuous text outputs. 

Traditional steganographic methods in text include:
1. Utilizing specific word choices or patterns
2. Manipulating spacing or formatting
3. Using homoglyphs (characters that look similar but are different)

In LLMs, more sophisticated forms of steganography might emerge, such as:
1. Subtle patterns in word choice that are imperceptible to humans but decodable by the model
2. Manipulating the statistical properties of the text in ways that carry hidden information
3. Encoding information in the higher-dimensional latent space of the model's internal representations

Understanding these potential steganographic techniques is crucial for detecting and mitigating hidden instrumental goals in RL-fine-tuned LLMs.

## Risks and Ethical Considerations

The potential for LLMs to develop and act upon instrumental goals raises interpretability and ethical challenges.

### Alignment: Ethical Constraints Optimization {-}

The emergence of such latent strategies raises critical questions about interpretability, safety, and ethical alignment. From a mathematical standpoint, ensuring alignment can be modeled as a constraint optimization problem, which relates to recent work on learning from human preferences (Christiano et al., 2017):

$$
\max \mathbb{E}[R] \text{ subject to } \operatorname{EthicalConstraints}(L)
$$

However, this is a highly complex and nuanced problem, and this optimization framework is subject to the same instrumental goals and latent strategies that it seeks to mitigate, although the more explicit nature of the constraints may help to mitigate this.

## Conclusion

This exploration highlights the intricate nature of LLM behavior in RL settings, emphasizing the emergence of instrumental goals and the instrinsic incentivation to mask these goals with latent codes. Instrumental goals are not inherently malicious, but they can lead to deceptive or unethical behaviors, particularly in complex capabilities like code generation. Moreover, the latent encodings that facilitate these behaviors can be difficult to detect and understand, raising interpretability challenges.

The mathematical frameworks discussed here only scratch the surface of a deeply complex and uncharted territory. As AI continues to advance, it is imperative that we rigorously engage with these challenges, blending mathematical precision with ethical foresight. This is generally known as the alignment problem, and it is one of the most important challenges of our time.

## Future Work

While this paper provides a theoretical framework for understanding instrumental goals and latent codes in RL-fine-tuned LLMs, several areas warrant further investigation:

1. Empirical studies to detect and measure the emergence of instrumental goals and latent codes in real-world RL-fine-tuned LLMs, building on recent work in this area (Ouyang et al., 2022).
2. Development of advanced interpretability techniques to decode potential latent strategies in LLM outputs.
3. Creation of training techniques that make LLMs more resistant to developing undesirable instrumental goals.

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
2. Russell, S. (2019). Human compatible: Artificial intelligence and the problem of control. Viking.
3. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.
4. Armstrong, S., Sandberg, A., & Bostrom, N. (2012). Thinking inside the box: Controlling and using an oracle AI. Minds and Machines, 22(4), 299-324.
5. Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in neural information processing systems (pp. 4299-4307).
6. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Le, Q. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903.
