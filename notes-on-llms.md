
If a model output has a single value, like a number, things like LLM consensus work quite well. If the output is, say, a proof, or a more complicated structure,
it can be hard to use consensus -- every outut is likely unique. Sometimes, we can use a grammar constraint to make it fit a certain syntax, in which case
this may make it more likely to produce more non-unique outputs, but I don't anticipate this working that great.

What else can we do?

1. Canonical forms. Map a high-dimensional output to a low-dimensional output by "regularizing" the grammar. In information retrieval, we have various ways of
   doing this, like `stemming`, `lower-case`, removing `stop words` like `the`, mapping synonymous words to some base form (which may be done before or after
   stemming, and so on. Essentially, we apply GOFAI techniques to make the outputs more consistent, and thus a better target for conesnsus-like algorithms.
   HOwever, it may also help in the case where we have, say, ways to verify an output. If the verification must only work over a low-dimensinoal form of the
   output, that may make it easier. Of course, we retain the high-dimensinal output and use that as the actual output.
