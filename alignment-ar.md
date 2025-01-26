# autoregressive LLM

suppose we have a sequence $x = x[t] x[t-1] ... x[1]$.
given a context length $n$, we can sample from
$P(X[t+1] = a | x)$ to generate a new sequence $a∥x$. rince and repeat.

## misalignment
we can use this to generate as large of a sequence as we want.
however, only some of these sequences will be "good", where
by "good" we mean aligned with what we humans find useful.
this is the AI alignment problem, which has a lot of big
philosophical ideas that we won't go into

suppose we have an aligned sequence $x$. then, if we
sample a new token $a$, what's the probability that
$a∥x$ is aligned? this is actually a difficult,
perhaps ill-posed problem. for instance, if we had an
aligned sequence
```
    "the cat in the
```
and then generate $a=hello$, then
```
    "the cat in the hello
```
seems misaligned. however, if we keep generating
tokens, it may become re-aligned, for instance,
```
    "the cat in the hello" is meaningless text.
```
there are many other such complications that
allow a sequence to go from aligned to misaligned and
back to aligned an a strictly autoregressive
model.

as a possibly over-simplifying assumption,
let's suppose that the probability of generating a token $a$
that causes an aligned sequence $x$ to become misaligned is
given by $e$, and that these errors are independent and we
consider the probability that it will go from misaligned to
aligned negligible. then, we see that the probability that generating
$n$ new tokens has a probability $(1-e)^n$ of still being aligned.
(or a probability $1-(1-e)^n$ of being misaligned).

this may or may not be a problem.

## symbol grounding
Symbol grounding offers a potential solution to the misalignment
issue. By incorporating feedback from an environment, LLMs can be
steered back towards generating aligned sequences.

### chatbots
humans conversing with a chatbot is a form of symbol grounding.
when the chatbot produces a misaligned response, the human
can provide feedback that may allow the chatbot to realign.
for instance, let's consider a variation of the earlier example
of misalignment:
```
    chatbot: the cat in the hello.
```
the human may respond with
```
    human: that's nonsense!
```
the new sequence
```
    chatbot: the cat in the hello.
    human: that's nonsense!
```
will now likely place the chatbot back into re-alignment.

### output from tools
LLMs that are integrated with tools can benefit from 
symbol grounding. for instance, if the LLM produces
python code and it's executed where the output, errors, and warnings are
fed back into the model, then we see that it can be realigned
in the same way the chatbot was re-aligned. (we might say that
in the chatbot example, the "tool" was human feedback.)

### self-chatter
We could have the LLM, or other LLMs, critique its outputs in
a separate context. this is similar to both tool use and chatbots.

## reinfrocement learning agent
moving beyond the autoregressive model, agentic architectures
may incorproate LLMs as a component within a larger reinforcement
learning agent. these agents have goals ranging from intrinsic
motivations (like curiosity) to explicit objectives (like
paperclip maximiation), where the LLM serves as an input rather
than the final output.

this architectue allows for better alignment with human values
while still leveraging the power of LLMs.
