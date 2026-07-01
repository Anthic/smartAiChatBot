# Studio Butterfly — AI Developer Take-Home Assignment
## Project Requirements & Scope Document

> This document describes **what needs to be built, why, and what is out of scope.** It contains no installation steps, code, or folder structure. Any AI or developer reading this should be able to understand exactly what the client wants.

---

## 1. What the Client Actually Wants (One-liner)

Build an **AI Assistant / Agent** that can:
- Read documents (PDF/TXT/MD) and answer questions from them (RAG)
- Remember conversation context (Memory)
- Call mock/fake tools when needed to fetch data (Tool Calling)
- **Decide on its own** which of the above is needed for each message (Routing/Decision Logic)

This is not a simple chatbot — it's a **decision-making agent**, whose job is to choose the correct path for every user message.

---

## 2. Deadline

**Saturday, 4 July 2026** — a public GitHub repo link must be submitted by replying to the assignment email before this date.

---

## 3. The Four Core Features (Per the Requirements)

### 3.1 Knowledge Ingestion
- The user uploads a PDF, TXT, or Markdown file
- The system must:
  - Load the document
  - Split it into chunks
  - Generate embeddings
  - Store the embeddings in a vector database (FAISS or ChromaDB)

### 3.2 Chat
- A chat API/UI where the user asks questions
- The assistant answers using the uploaded knowledge base
- If no answer is found, it must respond with **exactly** this sentence:
  > "I couldn't find that information in the uploaded documents."

### 3.3 Context Memory
- The assistant must remember conversation context within the current session
- Example 1:
  - User: "My name is John."
  - User: "What's my name?"
  - Assistant: "Your name is John."
- Example 2:
  - User: "I'm looking for a laptop."
  - User: "Show me cheaper options."
  - The assistant must understand that "cheaper options" refers to laptops

### 3.4 Tool Calling (Mock/Fake Tools — Not Real External APIs)

**Tool 1 — Order Status**
- Input: Order ID
- Output: Status + Estimated Delivery Date
- Data source: `orders.json` (sample provided in the assignment)
- Example: User asks "Where is my order ORD001?" → Assistant calls the tool and responds with the result

**Tool 2 — Product Search**
- Input: Product Name
- Output: Product Name, Price, Stock Availability
- Data source: `products.json` (sample provided in the assignment)
- Example: User asks "Do you have a wireless mouse?" → Assistant calls the tool and responds with the result

> **Important:** Both tools are entirely **mock/fake** — no real e-commerce backend or external API integration is required. A simple JSON file lookup is sufficient.

---

## 4. The AI Pipeline (How Everything Works Together)

The client specifically outlined this flow:

1. User uploads a document
2. Document is ingested (chunking → embeddings → vector database)
3. User sends a message
4. Conversation memory is loaded
5. The system decides whether the query requires:
   - Retrieval from the knowledge base, OR
   - A tool call, OR
   - A direct LLM response
6. Retrieval and/or tool call is executed if needed
7. The final response is generated using the LLM
8. The response is returned to the user

**Step 5 — the decision logic — is the most important part of this project.** This decision-making is exactly what makes it an "agent" rather than a hardcoded rule-based chatbot.

---

## 5. Deliverables

- [ ] Public GitHub repository
- [ ] README with setup instructions
- [ ] Sample `orders.json` and `products.json`
- [ ] Architecture/pipeline diagram (PNG, PDF, or Markdown/Mermaid)
- [ ] A brief written explanation covering:
  - How the ingestion pipeline works
  - The retrieval approach used
  - How memory is implemented
  - The tool-calling strategy
  - Prompt design decisions

---

## 6. Evaluation Criteria

- Knowledge ingestion and retrieval quality
- Chat functionality
- Whether context memory actually works
- Whether tool calling triggers correctly
- AI pipeline design (how smart the routing logic is)
- Code quality and project structure
- Clarity of documentation
- Error handling

---

## 7. Technical Constraints (Stated by the Client)

| Item | Requirement |
|---|---|
| Language | Python |
| Framework | FastAPI (preferred) |
| LLM | Any (OpenAI, Gemini, Ollama, etc.) |
| Embedding model | Any |
| Vector DB | FAISS or ChromaDB |

> **Mistral AI** has been chosen as the LLM for this project (has a free tier, native tool-calling support).

---



---

## 9. Core Philosophy (Stated Directly by the Client)

> "We're less interested in a perfect submission than in how you reason about the pipeline, structure your code, and document your decisions."

In other words — **clear reasoning, clean structure, and documented decisions matter more than perfect polish.** There's no need to over-engineer, but every pipeline decision (why this chunk size, why this retrieval approach, why this memory design) should be clearly explained in the README.

