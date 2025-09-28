# Contextualizer
Contextualizer: A Sampling Based Alternative to Attention Mechanism in Transformers 

Abdullah Nazhat Abdullah, Tarkan Aydin

Link: https://www.researchgate.net/publication/395695429_Contextualizer_A_Sampling_Based_Alternative_to_Attention_Mechanism_in_Transformers

# Abstract

The emergence of the transformer architecture signified many advances in natural language processing (NLP), and the pinnacle of these advances is represented by large language models (LLM). In addition, the transformer architecture has been employed by computer vision (CV) researchers and practitioners advancing image processing tasks and leading to the recent introduction of multi-task and multi-modal deep learning architectures. One drawback of the typical transformer architecture is its reliance on the scaled dot product attention mechanism in its design, resulting in high requirements for compute and memory and limiting the deployment of such designs. This paper presents a full experimental validation of a newly proposed mechanism substituting the attention mechanism incorporated in the typical transformer architecture on which a combination of downsampling, low parameter count representation transform, and upsampling are utilized, targeting the construction of a new computational block with a much reduced computational burden. An equalized experimental validation is performed in this work, showing that the proposed mechanism is highly competitive in comparison to baseline architectures. The obtained experimental results significantly support the hypothesis presented in this work, which replaces the typical attention mechanism with a linear projection or MLP representation transform applied on a reduced form of the input context through a process of sampling. This solution allows for implementing transformer architectures of suitable capability as well as low requirements to substitute the traditional attention mechanism in the design of transformer architectures.
