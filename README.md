### RAG Ingestion

Uploaded documents are parsed into plain text, split with `RecursiveCharacterTextSplitter`,
embedded with Mistral's `mistral-embed`, and stored in Qdrant.

Chunking uses `chunk_size=512` and `chunk_overlap=64`.

Reasoning:
- 512 characters keeps chunks focused and cheap to embed.
- 64 characters of overlap preserves context across chunk boundaries.
- This size works well for short support documents, PDFs, FAQs, and product docs.