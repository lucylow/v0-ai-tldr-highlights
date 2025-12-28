# v0-AI-TLDR-Highlights

**Streaming TL;DR + Smart Highlights — for Foru.ms & v0**
Repository: [https://github.com/lucylow/v0-ai-tldr-highlights](https://github.com/lucylow/v0-ai-tldr-highlights)

> Production-minded, judge-grade README for developers and evaluators. Contains architecture diagrams, deployment instructions, data model, ingestion & search code, training & evaluation guidance (including PerforatedAI integration), offline demo, and reproducible experiments (W&B sweep). Long, detailed, and technical — intended as the canonical developer doc.

---

> **Table of contents (jump links)**
>
> 1. [Project Summary](#project-summary)
> 2. [Quickstart (Dev)](#quickstart-dev)
> 3. [High-level Architecture](#high-level-architecture)
> 4. [Streaming Flow — sequence diagram](#streaming-flow---sequence-diagram)
> 5. [Database schema (pg + pgvector) & RLS](#database-schema-pg--pgvector--rls)
> 6. [Ingestion pipeline (Foru.ms → sentences → embeddings)](#ingestion-pipeline-forums-→-sentences-→-embeddings)
> 7. [Semantic search & ranking (pgvector + proxy + faiss fallback)](#semantic-search--ranking-pgvector--proxy--faiss-fallback)
> 8. [Streaming proxy & SSE forwarding (OpenAI / Anthropic)](#streaming-proxy--sse-forwarding-openai--anthropic)
> 9. [Frontend integration (Next.js / App Router)](#frontend-integration-nextjs--app-router)
> 10. [Model training & PerforatedAI integration](#model-training--perforatedai-integration)
> 11. [Evaluation & metrics (ROUGE, BERTScore, human checks)](#evaluation--metrics-rouge-bertscore-human-checks)
> 12. [W&B sweep — example YAML & protocol](#wandb-sweep---example-yaml--protocol)
> 13. [Docker demo & offline mode (FAISS)](#docker-demo--offline-mode-faiss)
> 14. [Security, privacy & operational concerns](#security-privacy--operational-concerns)
> 15. [Developer notes, tests & CI](#developer-notes-tests--ci)
> 16. [Roadmap & future work](#roadmap--future-work)
> 17. [Contributing & License](#contributing--license)

---

## Project summary

**v0-AI-TLDR-Highlights** is an end-to-end system that converts long forum threads into:

* a streaming TL;DR (token-by-token progressive output), and
* a ranked, sentence-level list of **Smart Highlights** with provenance (post ID + sentence index) and confidence.

It is designed to integrate with a headless forum backend (Foru.ms) and a modern frontend (v0 by Vercel / Next.js). The stack includes:

* Node.js / Express streaming proxy (for LLM streaming → SSE)
* Postgres + **pgvector** for semantic retrieval and provenance mapping (with Row-Level Security + multi-tenant support)
* Optional local FAISS fallback for offline demos
* PerforatedAI (dendritic optimization) integration for compressing models without losing accuracy
* W&B + sweep config for hyperparameter search and reproducible experiments

Goals: judge-friendly demo, measurable evaluation for model compression, production-viable architecture, and clear provenance to reduce hallucination risk.

---

## Quickstart (Dev)

> This assumes you have Node.js (18+), Docker, Python 3.10+, and Postgres installed. See each section below for detail.

### 1) Clone

```bash
git clone https://github.com/lucylow/v0-ai-tldr-highlights.git
cd v0-ai-tldr-highlights
```

### 2) Create `.env`

Copy `.env.example` -> `.env` and fill keys:

```env
# Server
PORT=8080
VECTOR_PROXY_SECRET=changeme

# Postgres
DATABASE_URL=postgres://postgres:postgres@localhost:5432/tldr

# OpenAI / Anthropic (for streaming & embeddings)
PROVIDER=openai
OPENAI_API_KEY=sk-...

# W&B (optional)
WANDB_API_KEY=...

# Local FAISS demo mode
VECTOR_MODE=pg  # or 'local' to use FAISS fallback
```

### 3) Start Postgres (local)

If using Docker Compose (provided `docker-compose.yml`):

```bash
docker compose up -d
```

Run DB migrations:

```bash
psql $DATABASE_URL -f migrations/001_init.sql
psql $DATABASE_URL -f migrations/002_core_tables.sql
psql $DATABASE_URL -f migrations/003_posts.sql
psql $DATABASE_URL -f migrations/004_sentences.sql
psql $DATABASE_URL -f migrations/005_rls.sql
psql $DATABASE_URL -f migrations/006_seed.sql
```

### 4) Install server deps & start

```bash
cd server
npm install
npm run dev
# default server: http://localhost:8080
```

### 5) Frontend

```bash
cd frontend
npm install
npm run dev
# default UI: http://localhost:3000
```

### 6) Optional local FAISS (Python)

```bash
cd local_faiss
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo_index.py
```

---

## High-level Architecture

Mermaid flowchart (renderable on GitHub if mermaid enabled):

```mermaid
flowchart LR
  subgraph Client
    A[Browser (Next.js)] --> |POST /stream_summary| B[Streaming Proxy (Node)]
    A --> |GET /thread/:id| C[Next.js Server APIs]
  end

  subgraph Backend
    B --> D[LLM Provider (OpenAI/Anthropic)]
    C --> E[Ingestion API]
    E --> F[Postgres + pgvector]
    F --> G[Vector Proxy (Edge safe)]
    G --> F
  end

  subgraph LocalDemo
    H[Local FAISS Index] -. fallback .-> G
  end

  D --> |final digest| B
  F --> |highlights| B
  B --> |SSE stream| A
```

**Key components:**

* **Frontend (Next.js + Tailwind):** thread viewer, summary panel, streaming UI, jump-to-sentence interactions.
* **Streaming Proxy (Node):** accepts frontend POST, forwards to LLM streaming API, parses provider stream, forwards SSE to client; calls highlights API post-consolidation.
* **Highlights API (Node):** ingestion endpoints, per-sentence embeddings, storage into Postgres `post_sentences` table (pgvector), ranking & retrieval endpoints.
* **Postgres + pgvector:** persistent vector store with IVFFlat index; Row-Level Security for multi-tenant separation.
* **Edge vector proxy:** thin server that accepts vector search requests from edge/runtime environments and queries Postgres on behalf of the edge.
* **Local FAISS fallback:** for offline demo, sentence embeddings are stored in an in-memory FAISS index accessible to the API.

---

## Streaming Flow — sequence diagram

```mermaid
sequenceDiagram
  participant Browser as Frontend
  participant Proxy as StreamingProxy
  participant LLM as LLMProvider
  participant API as HighlightsAPI
  participant DB as Postgres+pgvector

  Browser->>Proxy: POST /stream_summary { threadId, persona }
  Proxy->>API: GET /threads/{threadId} (server-side fetch from Foru.ms)
  API->>DB: fetch thread sentences (optional cache)
  API-->>Proxy: combined thread text
  Proxy->>LLM: streaming chat completion (stream=true)
  LLM-->>Proxy: stream tokens (data: {...})
  Proxy-->>Browser: SSE { type:'tl;dr_token', token }
  alt LLM streaming finished
    Proxy->>LLM: non-stream consolidation call
    LLM-->>Proxy: final digest (bullets)
    Proxy->>API: POST /highlights { threadId, digest_text }
    API->>DB: compute embedding(digest_text) + search top sentences
    DB-->>API: highlight rows (postId, sentenceIdx, text, score)
    API-->>Proxy: highlights
    Proxy-->>Browser: SSE { type:'digest', bullets, highlights }
  end
```

**Notes:**

* The streaming proxy implements backpressure handling (`res.write` drain) and safe aborts on client disconnect.
* Consolidation call is used to produce high-quality final digest; highlights are computed from the digest + stored sentence embeddings to ensure provenance.

---

## Database schema (pg + pgvector + RLS)

Below are production-ready SQL migrations & rationale.

### 1. Enable extensions

`migrations/001_init.sql`

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid()
```

### 2. Tenants table (multi-tenant)

`migrations/002_core_tables.sql`

```sql
CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slug TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);
```

### 3. Threads & posts (mirror Foru.ms)

`migrations/003_posts.sql`

```sql
CREATE TABLE forum_threads (
  id TEXT PRIMARY KEY,
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  title TEXT,
  created_at TIMESTAMP
);

CREATE TABLE forum_posts (
  id TEXT PRIMARY KEY,
  thread_id TEXT REFERENCES forum_threads(id) ON DELETE CASCADE,
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  author TEXT,
  content TEXT,
  votes INTEGER DEFAULT 0,
  created_at TIMESTAMP
);
```

### 4. Sentence table (searchable vector)

`migrations/004_sentences.sql`

```sql
CREATE TABLE post_sentences (
  id BIGSERIAL PRIMARY KEY,
  tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
  thread_id TEXT,
  post_id TEXT,
  sentence_index INTEGER,
  sentence TEXT NOT NULL,
  embedding VECTOR(1536), -- adjust dim to embedding model
  confidence FLOAT,
  importance FLOAT,
  created_at TIMESTAMP DEFAULT now()
);
-- IVF index for cosine
CREATE INDEX idx_sentence_embedding
  ON post_sentences
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

> **Why vector_dim=1536?** This is the dim for OpenAI `text-embedding-3-small`. If you use a different embedding model (e.g., `all-MiniLM-L6-v2` dim=384), adapt column type accordingly (VECTOR(384)).

### 5. RLS policies

`migrations/005_rls.sql`

```sql
ALTER TABLE forum_threads ENABLE ROW LEVEL SECURITY;
ALTER TABLE forum_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_sentences ENABLE ROW LEVEL SECURITY;

-- policy: allow only rows matching session tenant
CREATE POLICY tenant_threads_policy
  ON forum_threads
  USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_posts_policy
  ON forum_posts
  USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_sentences_policy
  ON post_sentences
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

### 6. Seed data for demo

`migrations/006_seed.sql` (example)

```sql
INSERT INTO tenants (id, slug, name) VALUES
('11111111-1111-1111-1111-111111111111','demo','Demo Community');

INSERT INTO forum_threads (id, tenant_id, title, created_at)
VALUES ('thread_demo_1','11111111-1111-1111-1111-111111111111','Demo: Streaming Summary', now());

INSERT INTO forum_posts (id, thread_id, tenant_id, author, content, votes, created_at)
VALUES
('post_demo_1','thread_demo_1','11111111-1111-1111-1111-111111111111','alice','We should chunk the docs and stream partial summaries.', 12, now()),
('post_demo_2','thread_demo_1','11111111-1111-1111-1111-111111111111','bob','Cache embeddings per sentence to speed up re-runs.', 6, now());
```

---

## Ingestion pipeline (Foru.ms → sentences → embeddings → pgvector)

High level steps:

1. fetch thread & posts from Foru.ms API (server side)
2. sanitize and split each post into sentences (preserve offsets)
3. compute sentence embeddings (OpenAI / sentence-transformers)
4. insert rows into `post_sentences` with `(tenant_id, thread_id, post_id, sentence_index, sentence, embedding)`

### Example ingestion code (Node/TypeScript)

`server/ingest/ingestThread.ts` (simplified)

```ts
import { pool, withTenant } from '../db';
import fetchThreadData from '../foru/fetchThread';

function splitSentences(text: string) {
  // Basic sentence splitter; replace with NLP pipeline for production
  return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
}

async function embedTexts(texts: string[]): Promise<number[][]> {
  // call OpenAI embeddings API (or local model)
  const res = await fetch('https://api.openai.com/v1/embeddings', {...});
  // parse and return array of arrays
}

export async function ingestThread(tenantId: string, threadId: string) {
  const thread = await fetchThreadData(threadId);
  await withTenant(pool, tenantId, async (db) => {
    // upsert thread
    await db.query(`INSERT INTO forum_threads (id, tenant_id, title, created_at) VALUES ($1,$2,$3,$4) ON CONFLICT (id) DO NOTHING`, [thread.id, tenantId, thread.title, thread.created_at]);
    for (const post of thread.posts) {
      await db.query(`INSERT INTO forum_posts (id, thread_id, tenant_id, author, content, votes, created_at) VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (id) DO NOTHING`, [post.id, thread.id, tenantId, post.author, post.content, post.votes, post.created_at]);
      const sentences = splitSentences(post.content);
      if (!sentences.length) continue;
      const embeddings = await embedTexts(sentences);
      for (let i=0;i<sentences.length;i++){
        await db.query(`INSERT INTO post_sentences (tenant_id, thread_id, post_id, sentence_index, sentence, embedding, confidence, importance) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, [tenantId, thread.id, post.id, i, sentences[i], embeddings[i], null, null]);
      }
    }
  });
}
```

**Notes:**

* Embedding batches should be limited (e.g., chunks of 50) to avoid rate limits.
* Sanitize and optionally remove code blocks or binary artifacts before embedding.

---

## Semantic search & ranking (pgvector + proxy + FAISS fallback)

This section explains how to use pgvector for nearest neighbor search, an edge-proxy for serverless environments, and a FAISS fallback for offline/low-resource demos.

### Querying pgvector in Postgres (cosine)

Search for most similar sentences to a query embedding (Postgres syntax using `<=>` operator — lower is better for distance):

```sql
SELECT id, sentence, post_id, sentence_index, 1 - (embedding <=> $1) AS similarity
FROM post_sentences
WHERE thread_id = $2
ORDER BY embedding <=> $1
LIMIT 10;
```

> `embedding <=> $1` is the vector_distance operator for ivfflat / vector_cosine_ops.

### Edge-compatible proxy (Vercel/E2)

Edge runtimes cannot open raw TCP to Postgres. Use a lightweight proxy endpoint hosted in Node (not on edge) that performs the vector search. Edge functions can call the proxy via https. The proxy requires a secret and enforces tenant RLS via `SET app.tenant_id`.

`proxy/vector-search.ts`

```ts
app.post('/vector_search', async (req, res) => {
  const { threadId, embedding, tenantId } = req.body;
  // set session tenant for RLS
  await client.query('SET app.tenant_id = $1', [tenantId]);
  const rows = await client.query('SELECT sentence, post_id, sentence_index, 1 - (embedding <=> $1) AS similarity FROM post_sentences WHERE thread_id = $2 ORDER BY embedding <=> $1 LIMIT 10', [embedding, threadId]);
  res.json(rows.rows);
});
```

Edge route (Next.js route.ts on edge) POSTs to proxy with secret header.

### Local FAISS fallback (Python)

When `VECTOR_MODE=local`, the API uses FAISS in Python (or a local Node binding) to perform vector search.

`local/faiss_index.py` (Python)

```py
import faiss, numpy as np
class LocalIndex:
  def __init__(self, dim):
    self.dim = dim
    self.index = faiss.IndexFlatIP(dim)
    self.meta = []  # metadata per vector

  def add(self, embeddings, metas):
    self.index.add(np.array(embeddings).astype('float32'))
    self.meta.extend(metas)

  def search(self, qemb, k=10):
    D, I = self.index.search(np.array([qemb]).astype('float32'), k)
    return [{'meta': self.meta[i], 'score': float(D[0][j])} for j,i in enumerate(I[0])]
```

Switch logic in Node:

```ts
if (process.env.VECTOR_MODE === 'local') {
  // call local vector server
} else {
  // call pgvector proxy
}
```

---

## Streaming proxy & SSE forwarding (OpenAI / Anthropic)

This is a core piece: it forwards LLM streaming payloads to the browser as SSE and handles consolidation + provenance retrieval.

### Goals:

* Accept POST from frontend: `{ threadId, persona }`
* Fetch thread data (server side) and send to LLM streaming endpoint
* Parse provider stream (OpenAI: `data: {...}` lines) and forward tokens to browser as `data: { type:'tl;dr_token', token }` SSE events
* After streaming, call a consolidation (non-streaming) LLM call to produce final digest bullets
* Compute highlights via Highlights API (embedding digest -> query -> pgvector search)
* Send a final `digest` SSE event: `{ bullets, highlights }`
* Handle backpressure (`res.write` return false) by awaiting `'drain'` event
* Retry provider calls on intermittent failures with exponential backoff
* Close SSE on client disconnect

### Simplified implementation (Node/Express)

`server/streaming-proxy.js`

```js
app.post('/stream_summary', async (req, res) => {
  const { threadId, persona } = req.body;
  // set SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection','keep-alive');
  res.flushHeaders();

  // fetch thread & posts server-side
  const thread = await fetchThreadData(threadId);
  const combinedText = threadToText(thread);

  // streaming call to provider
  const stream = await callOpenAIStreaming(combinedText, persona);

  try {
    for await (const chunk of stream) {
      // chunk example: { token: 'the', done: false }
      const ok = res.write(`data: ${JSON.stringify({ type:'tl;dr_token', token: chunk.token })}\n\n`);
      if (!ok) await once(res, 'drain'); // backpressure
      if (chunk.done) break;
    }

    // consolidation call
    const bullets = await callOpenAInonStreaming(`Consolidate: ${combinedText}`, persona);
    const digest_embedding = await embedText(bullets.join('\n'));
    const highlights = await fetchHighlights(threadId, digest_embedding);
    res.write(`data: ${JSON.stringify({ type:'digest', bullets, highlights })}\n\n`);
    res.end();
  } catch (err) {
    res.write(`data: ${JSON.stringify({ type:'error', message: err.message })}\n\n`);
    res.end();
  }
});
```

**Notes:**

* `callOpenAIStreaming` should parse the provider stream and yield tokens in an async iterator.
* For Anthropic provider, adjust parsing accordingly.
* `fetchHighlights` calls the Highlights API or vector proxy to perform embedding search.

---

## Frontend integration (Next.js / App Router)

Key pages/components:

* `app/thread/[id]/page.tsx` (server component): pre-fetch thread data (server side), split into sentences for anchors.
* `components/ThreadPageClient.tsx` (client component): UI, starts streaming via `useStreamSummary`.
* `hooks/useStreamSummary.tsx`: POST to `/api/stream_summary` and read streaming body with ReadableStream; parse SSE/generic chunks and call callbacks (`onToken`, `onDigest`, `onHighlight`).
* `components/ImprovedSummaryPanel.tsx`: polished UI with TokenTicker, digest, highlights, copy/share buttons, persona selectors.
* `components/PostList.tsx`: renders posts with sentence anchors `id="post-<postId>-sent-<idx>"`. On highlight click, scrolls into view and pulses.

### Example client streaming hook (simplified)

`hooks/useStreamSummary.ts`

```ts
export default function useStreamSummary() {
  const controllerRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  async function startStream(opts) {
    const res = await fetch('/api/stream_summary', { method:'POST', body: JSON.stringify(opts) });
    if (!res.ok || !res.body) throw new Error('No body');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    setIsStreaming(true);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        const payload = part.startsWith('data: ') ? part.slice(6) : part;
        try {
          const obj = JSON.parse(payload);
          if (obj.type === 'tl;dr_token') opts.onToken(obj.token);
          else if (obj.type === 'digest') opts.onDigest(obj.bullets, obj.highlights);
        } catch (e) {
          opts.onToken(part);
        }
      }
    }
    setIsStreaming(false);
  }

  function stopStream() {
    controllerRef.current?.abort();
  }

  return { startStream, stopStream, isStreaming };
}
```

---

## Model training & PerforatedAI integration

We provide a training template `train.py` that supports:

* dataset loading (SAMSum, Reddit TL;DR, CNN/DailyMail)
* wrapper helpers to wrap T5 and SBERT blocks for PerforatedAI (`pai_wrappers.py`)
* PerforatedBackPropagationTracker integration pattern
* W&B logging hooks

### Experiment protocol (3-experiment)

Run these 3 scenarios to isolate dendritic optimization benefits:

1. **Baseline** — original model (e.g., `t5-small` or `all-MiniLM`) fine-tuned.
2. **Compressed + Dendrites** — compress architecture (fewer layers / smaller d_model) then run PerforatedAI to add dendrites during training.
3. **Compressed control** — same compressed architecture trained without PerforatedAI.

Track metrics:

* ROUGE-1/2/L, BERTScore for summarization
* Highlight precision (top-k relevance)
* Params, FLOPs, inference latency, GPU memory

### `pai_wrappers.py` (example)

```py
class _ModuleWrapper(nn.Module):
  def __init__(self, orig): super().__init__(); self.orig = orig
  def forward(self, *args, **kwargs): return self.orig(*args, **kwargs)

def wrap_t5_layers_for_pai(model):
  for i, blk in enumerate(model.encoder.block):
    model.encoder.block[i] = _ModuleWrapper(blk)
  for i, blk in enumerate(model.decoder.block):
    model.decoder.block[i] = _ModuleWrapper(blk)
  return model
```

### Training loop skeleton (PerforatedAI)

```py
tracker = PA.PerforatedBackPropagationTracker(do_pb=True, save_name="t5_tldr")
tracker.setup_optimizer(AdamW, torch.optim.lr_scheduler.StepLR, lr=2e-5)
while not training_complete:
  train_one_epoch()
  val_metric = evaluate()
  returned = tracker.add_validation_score(model, val_metric)
  if restructured:
    tracker.setup_optimizer(...)
```

**Important:** always run the three experiments and publish the numeric table. Aim for ≥15–30% parameter reduction with equal or better ROUGE/BERTScore.

---

## Evaluation & metrics (ROUGE, BERTScore, human checks)

### Automatic metrics

* **ROUGE-1 / ROUGE-2 / ROUGE-L** — classic summarization metrics (sentence-level & n-gram overlap)
* **BERTScore** — semantic similarity measure
* **Embedding Cosine** between generated digest and reference summary
* **Highlight Precision@k** — % of top-k highlights judged relevant automatically (if you have ground truth).

### Human eval

* **Faithfulness**: randomly sample 20 final digests, ask raters to score 1–5 for factual accuracy vs thread.
* **Highlight relevance**: raters evaluate whether each highlight accurately reflects an important claim/answer (binary or Likert).

### Demo success criteria (hackathon)

* Streaming first token < 500ms perceived (depends on network)
* Final digest produced in < 10s for threads ≤ 2000 tokens (with cheap streaming draft + consolidation)
* Achieve ROUGE comparable to baseline with compressed + dendrites experiment or show meaningful param reduction with similar BERTScore

---

## W&B sweep — example YAML & protocol

`configs/sweep_encoder_t5.yaml` (Bayesian)

```yaml
program: train.py
method: bayes
metric:
  name: val_rouge_l
  goal: maximize
parameters:
  batch_size: {values: [8,16,32]}
  lr: {min: 1e-6, max: 5e-4}
  enc_num_layers: {values: [4,6,8]}
  enc_hidden_size: {values: [256,384,512]}
  dec_num_layers: {values: [3,4,6]}
  dec_d_model: {values: [256,384,512]}
  do_pb: {values: [true,false]}
  N_EPOCHS_TO_SWITCH: {values: [3,5,10]}
  P_EPOCHS_TO_SWITCH: {values: [1,3,5]}
```

**Protocol:**

* Run baseline tune to set sensible LR/WD.
* Run sweep comparing `do_pb=true/false` across compression settings.
* Log `num_params`, `inference_latency` as custom metrics.
* Publish W&B links in Devpost.

---

## Docker demo & offline mode (FAISS)

We provide a `Dockerfile.demo` and `docker-compose.demo.yml` which start:

* Postgres with pgvector (optional)
* streaming proxy (Node) in demo mode
* mock LLM server (`mock_llm_server.py`) that streams tokens via SSE for UI demos
* local FAISS server (Python) if `VECTOR_MODE=local`

`docker-compose.demo.yml` sample:

```yaml
version: '3.8'
services:
  db:
    image: postgis/postgis:14-3.3
    environment:
      POSTGRES_PASSWORD: postgres
    ports: ["5432:5432"]
  api:
    build: ./server
    env_file: .env
    ports: ["8080:8080"]
  mockllm:
    build: ./mock_llm
    ports: ["9000:9000"]
```

Run demo:

```bash
docker-compose -f docker-compose.demo.yml up --build
# open http://localhost:3000 (frontend)
```

---

## Security, privacy & operational concerns

**API keys & secrets**

* Never expose OpenAI/Anthropic keys to client. Use server proxy and environment variables.
* For Edge calls use a proxy with `VECTOR_PROXY_SECRET` header.

**RLS & multi-tenant isolation**

* RLS enforces tenant boundaries via `SET app.tenant_id` session variable on DB connection.
* Always set session tenant on connection before running queries (use `withTenant()` pattern).

**Access control**

* Protect ingestion endpoints and vector proxy endpoints with authentication and rate limits, especially in multi-tenant deployments.

**Data retention & privacy**

* Optionally support redaction and TTL for embeddings for privacy compliance.
* Consider PII detection and removal during ingestion.

**Cost control**

* Cache embeddings and digests; do not re-embed sentences unnecessarily.
* Use cheaper embedding models for large-scale ingestion; use higher-quality models for consolidation and human-facing digests.

---

## Developer notes, tests & CI

**Unit & integration tests**

* Unit test sentence splitter, DB queries, embedding marshalling.
* Integration tests for vector search endpoints using in-memory FAISS server and a small seed dataset.

**CI**

* Use GitHub Actions:

  * lint & tests on PR
  * integration job runs with Docker Compose (db + server + mock LLM)
  * deploy job for `main` to staging environment (Vercel + node server on Railway)

**Key scripts**

* `server/scripts/run_migrations.sh` — apply migrations
* `server/scripts/seed_demo.sh` — seed demo data & build FAISS index (if using local mode)
* `frontend/scripts/start_local.sh` — start dev server with `.env`

---

## Roadmap & future work

* Optimize vector index with HNSW or PGVector quantization for larger corpora.
* Add per-sentence salience model to compute `importance`.
* Add user feedback loop: let users endorse highlights and use signals to fine-tune ranking.
* Add multilingual summarization & embeddings.
* Add conversational summarization (thread aggregation with temporal weighting).
* Integrate with commercial vector DBs (Pinecone, Weaviate) and offer hybrid BM25 + vector ranking.

---

## Contributing & License

We welcome contributions:

* Open an issue for major changes or feature requests
* PRs: include tests and update docs
* Maintain coding standards (prettier, eslint, black for Python)

**License:** MIT

---

## Appendix A — Example API endpoints

**Streaming**

```
POST /stream_summary
body: { threadId: string, persona: 'novice'|'technical'|'exec' }
SSE events:
  data: {"type":"tl;dr_token","token":"..."}
  data: {"type":"digest","bullets":[...],"highlights":[{postId, sentenceIdx, text, confidence}]}
```

**Ingest**

```
POST /ingest_thread
body: { tenantId, threadId }
returns: { ok: true }
```

**Highlights search (proxy)**

```
POST /vector_search
body: { tenantId, threadId, query_embedding }
returns: { highlights: [{sentence, post_id, sentence_index, similarity}] }
```

**Query thread**

```
GET /api/foruThread?threadId=...
returns: { thread, posts:[{postId, author, content, sentences:[{idx,text}]}] }
```

---

## Appendix B — Troubleshooting

* **SSE connection drops**: ensure `res.flushHeaders()` is called and server keeps connection alive. Use `Connection: keep-alive`.
* **pgvector slow on cold start**: increase `lists` param in IVFFlat index and run `ANALYZE` after large inserts.
* **Embedding rate limits**: throttle with batching and exponential backoff.

---

## Appendix C — Example: Full end-to-end demo workflow (commands)

1. Start demo stack:

```bash
docker compose -f docker-compose.demo.yml up --build
```

2. Seed demo:

```bash
curl -X POST http://localhost:8080/ingest_thread -H 'Content-Type: application/json' -d '{"tenantId":"11111111-1111-1111-1111-111111111111", "threadId":"thread_demo_1"}'
```

3. Open UI: `http://localhost:3000/thread/thread_demo_1` and click **Stream TL;DR**.

---

## Technical diagrams (mermaid)

### Architecture diagram

```mermaid
flowchart LR
  browser[Browser (Next.js UI)]
  proxy[Streaming Proxy]
  highlights[Highlights API]
  postgres[Postgres + pgvector]
  llm[LLM Provider]
  faiss[Local FAISS (fallback)]

  browser -->|POST /stream_summary| proxy
  proxy -->|fetch thread| highlights
  highlights -->|read/write| postgres
  highlights -->|vector search| postgres
  proxy -->|streaming API| llm
  proxy -->|SSE tokens| browser
  postgres -->|fallback| faiss
```

### Sequence: Ingest & highlight retrieval

```mermaid
sequenceDiagram
  participant Foru as Foru.ms API
  participant Ingest as Ingest Service
  participant DB as Postgres+pgvector
  participant Embed as Embedding Service

  Foru->>Ingest: GET thread posts
  Ingest->>Ingest: split sentences
  Ingest->>Embed: batch embed(sentences)
  Embed-->>Ingest: embedding arrays
  Ingest->>DB: INSERT post_sentences (embedding)
  DB-->>Ingest: OK
```
