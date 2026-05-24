# Attention Is All You Need
### [Paper Link](https://arxiv.org/pdf/1706.03762)

Previously for Language Modeling tasks, the dominant architectures were recurrence based (RNNs, GRUs, LSTMs), which process tokens sequentially.
The limitation was that sequential processing can't be parallelized across timesteps (you need state t to compute state t+1), which made training slow.
Long range dependency was also problematic.

## Attention before the Transformer
The authors didn't invent the Attention mechanism, they invented using it ALONE. They drop the RNN part entirely and just rely on tokens attending to each other to pass on information and of course also encoding token position through sinusoidal positional embeddings.

## The Attention Mechanism
The attention mechanism is a means for information to pass across token vectors. 

Let's assume a trained token embedding model. The word "the"'s vector doesn't semantically encode much by itself. The meaning of "the" depends heavily on it's context. Language in general is extremely contextual and that means information needs to mix.

If vector embeddings are solely independent before going through the attention mechanism, afterwards each embedding becomes more contextual. 

Let's learn how attention works with this example:  
*"the quick brown fox jumps over the lazy dog"*

Let's assume we tokenize by word.  
Each word here becomes a vector of some dimension.  
This vector encodes the meaning of the word by itself.  


<sub>-igh im tired now ima finish this writeup later</sub>





