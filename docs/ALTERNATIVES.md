# Alternative Tools And Best Use Cases

The current project uses simple local tools so you can learn the RAG pipeline without extra setup.

This file explains what we used, why we used it, and what to use later.

## Pipeline Orchestration

Current choice: small custom Python functions.

Why:

- easiest to understand
- no framework magic
- each RAG stage is visible
- good for learning fundamentals

Alternatives:

| Tool | Best use case | Notes |
| --- | --- | --- |
| LangChain | Building apps that combine many model calls, tools, retrievers, agents, and chains | Popular and flexible, but can feel abstract for beginners |
| LlamaIndex | Document-heavy RAG apps with many ingestion and indexing patterns | Excellent for RAG, especially when data connectors matter |
| Haystack | Production search and QA pipelines | Strong when you want more traditional pipeline components |
| Custom Python | Learning, prototypes, or highly controlled production code | More work later, but easiest to reason about |

Recommended path:

Learn with custom Python first. Try LlamaIndex next if your project is mostly document search. Try LangChain next if your app needs tools, agents, or complex workflows.

## Document Loading

Current choice: built-in text reading plus optional `pypdf`.

Why:

- markdown and text are simple
- `pypdf` is lightweight for basic PDFs
- page metadata is easy to preserve

Alternatives:

| Tool | Best use case | Notes |
| --- | --- | --- |
| `pypdf` | Simple digital PDFs | Good first PDF parser, weak for complex layouts |
| PyMuPDF | PDFs with layout, images, or faster extraction needs | Often better extraction quality than basic PDF tools |
| Unstructured | Mixed files such as PDF, HTML, DOCX, PPTX, email | Useful for production ingestion, heavier dependency stack |
| Docling | Rich document parsing and structure extraction | Good when tables and layout matter |
| Tesseract OCR | Scanned PDFs and images | Needed when text is not digitally embedded |
| Apache Tika | Enterprise document extraction across many file types | Useful in Java-heavy or enterprise environments |

Recommended path:

Use markdown or text while learning. Use PyMuPDF or Unstructured when your PDFs become messy. Add OCR when documents are scanned.

## Chunking Strategy

Current choice: word-count chunking with overlap.

Why:

- very easy to understand
- deterministic
- no tokenizer dependency
- shows why chunk size and overlap matter

Alternatives:

| Strategy | Best use case | Notes |
| --- | --- | --- |
| Fixed word chunks | Learning and quick prototypes | Simple but can split sections awkwardly |
| Token chunks | LLM-aware chunk sizing | Better when context limits matter |
| Sentence chunks | Clean readable passages | Good for FAQs and short text |
| Paragraph chunks | Articles, policies, and documentation | Preserves natural meaning units |
| Markdown header chunks | Technical docs and course notes | Keeps section hierarchy |
| Semantic chunking | Long complex documents | Uses embeddings to split by meaning |
| Parent-child chunks | Need precise retrieval plus wider context | Retrieve small chunks, send larger parent context |

Recommended path:

Start with word chunks. Move to markdown or paragraph chunking for notes and documentation. Use parent-child chunks when answers need more surrounding context.

## Embeddings

Current choice: local feature-hashing embedder.

Why:

- no API key
- no model download
- works offline
- teaches vector search clearly

Limitations:

- not truly semantic
- mostly catches word and phrase overlap
- weaker when the question uses different words than the document

Alternatives:

| Tool or model family | Best use case | Notes |
| --- | --- | --- |
| Sentence Transformers | Local neural embeddings | Great next step for learning real semantic search |
| OpenAI embeddings | High-quality hosted embeddings | Good quality and simple API, requires API key |
| Cohere embeddings | Search-focused hosted embeddings | Strong for enterprise retrieval |
| Voyage embeddings | High-quality retrieval use cases | Often strong in RAG benchmarks |
| BGE / E5 models | Open-source embedding models | Good local or self-hosted choices |
| Ollama embeddings | Local embeddings with simple setup | Convenient if already using Ollama |

Recommended path:

Replace `HashingEmbedder` with Sentence Transformers first. Later try a hosted embedding API and compare retrieval quality.

## Vector Storage

Current choice: JSON file with linear search.

Why:

- transparent
- no database setup
- easy to inspect
- perfect for tiny learning datasets

Limitations:

- slow for large indexes
- no concurrent writes
- no filtering engine beyond custom code
- not production storage

Alternatives:

| Tool | Best use case | Notes |
| --- | --- | --- |
| FAISS | Fast local similarity search | Great for local experiments and large in-memory indexes |
| ChromaDB | Beginner-friendly local vector database | Excellent next step after JSON |
| Qdrant | Production vector database with filtering | Strong open-source option with good metadata filtering |
| Weaviate | Production semantic search platform | Good when schema and hybrid search matter |
| Milvus | Very large-scale vector search | Strong for high-scale deployments |
| PostgreSQL + pgvector | Apps already using Postgres | Great when you want relational data and vectors together |
| Elasticsearch / OpenSearch | Hybrid keyword and vector search | Strong when keyword search is still important |

Recommended path:

Use ChromaDB for the next learning upgrade. Use Qdrant or pgvector when building a real app.

## Retrieval

Current choice: cosine similarity over stored vectors.

Why:

- standard retrieval baseline
- easy to explain
- works with normalized vectors

Alternatives:

| Method | Best use case | Notes |
| --- | --- | --- |
| Vector search | Semantic similarity | Best general starting point |
| Keyword search | Exact names, codes, IDs, and rare terms | Often beats vector search for exact terms |
| Hybrid search | Real-world document search | Combines keyword and vector strengths |
| Metadata filtering | Search within one file, user, date, product, or category | Essential for multi-tenant or large apps |
| Reranking | Improve top results after initial retrieval | Very useful when top-k has noisy matches |
| Query rewriting | Vague or conversational questions | Rewrites user question into a better search query |
| Multi-query retrieval | Questions that can be phrased many ways | Retrieves from multiple generated queries |

Recommended path:

Add metadata filtering first, then hybrid search, then reranking.

## Generation

Current choice: extractive answerer by default, optional Ollama.

Why extractive:

- runs everywhere
- does not hallucinate new wording as much
- shows retrieval quality clearly

Why Ollama optional:

- allows local LLM generation
- no cloud API key required
- easy to turn on with environment variables

Alternatives:

| Generator | Best use case | Notes |
| --- | --- | --- |
| Extractive answerer | Debugging retrieval | Not very fluent, but transparent |
| Ollama | Local experimentation | Good if your laptop can run the model |
| OpenAI API | High-quality hosted generation | Strong answer quality and tool ecosystem |
| Anthropic API | Long-context reasoning and writing | Good for careful document reasoning |
| Gemini API | Google ecosystem and multimodal workflows | Useful for mixed media and long context |
| vLLM | Self-hosted high-throughput LLM serving | Good for production teams with GPUs |

Recommended path:

Use the extractive answerer while debugging retrieval. Use Ollama or a hosted LLM when you want natural final answers.

## Prompting

Current choice: a grounded prompt that says answer only from context.

Why:

- reduces hallucination
- makes missing information behavior explicit
- encourages source citations

Alternatives:

| Prompt pattern | Best use case | Notes |
| --- | --- | --- |
| Basic grounded prompt | First RAG app | Simple and effective |
| Structured JSON output | APIs and automation | Easier for downstream code to parse |
| Citation-enforced prompt | Research, legal, policy, support | Forces answer-source mapping |
| Chain-of-thought style private reasoning | Complex reasoning | Use carefully; final answer should stay concise |
| Refusal-aware prompt | Compliance and high-stakes domains | Teaches model when not to answer |

Recommended path:

Start with the grounded prompt. Add structured outputs when building a UI or API.

## User Interface

Current choice: command-line interface.

Why:

- fastest to build
- easiest to debug
- no frontend framework required

Alternatives:

| UI | Best use case | Notes |
| --- | --- | --- |
| CLI | Learning and debugging | Best for early development |
| Streamlit | Quick Python web apps | Great next step for a chat UI |
| Gradio | ML demos and sharing | Very fast for model demos |
| FastAPI | Backend API | Good when building a real service |
| React / Next.js | Production web app | Best for polished user experience |

Recommended path:

Add Streamlit after you are comfortable with the CLI.

## Evaluation

Current choice: small unit tests.

Why:

- checks core behavior
- fast feedback
- easy for beginners

Alternatives:

| Evaluation method | Best use case | Notes |
| --- | --- | --- |
| Unit tests | Code correctness | Good for chunking, loading, formatting |
| Golden Q&A set | RAG answer quality | Ask known questions and compare expected answers |
| Retrieval metrics | Retriever quality | Recall@k and precision@k are useful |
| LLM-as-judge | Larger answer evaluation | Useful but needs careful prompts and sampling |
| RAGAS / DeepEval | Automated RAG evaluation | Good once the project grows |
| Human review | High-stakes applications | Still important for production |

Recommended path:

Create 10 known questions from your own documents. Track whether the right chunk is retrieved before judging answer quality.

## Best Upgrade Sequence

If you want to keep learning step by step, upgrade in this order:

1. Add your own markdown notes.
2. Use `--show-context` to inspect retrieval.
3. Replace the embedder with Sentence Transformers.
4. Replace JSON storage with ChromaDB.
5. Add a Streamlit chat UI.
6. Add metadata filters.
7. Add reranking.
8. Add evaluation questions.
9. Add a hosted or local LLM for fluent answers.

This order keeps the learning curve smooth because each upgrade changes only one part of the RAG pipeline.

