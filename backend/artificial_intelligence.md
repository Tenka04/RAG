# Overview of Artificial Intelligence and Machine Learning

## What is Artificial Intelligence?
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. The term is frequently applied to the project of developing systems endowed with intellectual processes characteristic of humans, such as the ability to reason, discover meaning, generalize, or learn from past experience.

## Generative AI and Large Language Models
Generative Artificial Intelligence is an umbrella term for machine learning models capable of generating novel text, imagery, audio, and code in response to user prompts.
Large Language Models (LLMs), such as LLaMA, GPT, and Gemma, are trained on vast corpora of textual data using Transformer neural network architectures. They predict subsequent tokens in a sequence, allowing them to comprehend complex semantics and generate natural language.

## Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances LLM accuracy and reliability by dynamically retrieving relevant facts from an authoritative external knowledge base before generating a response.
The standard RAG pipeline consists of five key phases:
1. **Document Loading**: Ingesting unstructured text from various file formats (PDF, DOCX, Markdown, Text).
2. **Chunking**: Splitting raw documents into manageable semantic units with consistent overlap to preserve contextual continuity across boundaries.
3. **Embedding**: Converting text chunks into dense mathematical vector representations using neural embedding models.
4. **Vector Indexing & Retrieval**: Storing vector embeddings in a high-performance vector store (such as ChromaDB) and executing cosine distance or inner product similarity searches against user queries.
5. **Generation**: Injecting top retrieved context chunks into the prompt context window of an LLM to generate grounded, factually verifiable responses with source citations.
