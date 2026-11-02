"""Hand-written narrative for evaluation/REPORT.md: definitions, a frozen historical
baseline, and known limitations.

Kept separate from report.py so that re-running the evaluation - which only recomputes
the numeric sections (tables, scores, latencies) - can never silently drop this text.
Only true constants and templates filled from local facts belong here; anything that
needs a fresh DIAL call stays computed in report.py.
"""

SETUP_TEMPLATE = """\
- LLM: `{chat_model}` (DIAL / Azure OpenAI-compatible), temperature 0.
- Embedding model: `{embedding_model}`.
- Chunking: {chunk_size} characters per chunk, {chunk_overlap} characters overlap (`RecursiveCharacterTextSplitter`).
- Vector database: Chroma, collection `{collection_name}`, persisted locally.
- Retrieval: top-K = {top_k} (`config.TOP_K`) — the same K the live chatbot uses for `search_parking_info`.
- Corpus: {doc_count} static markdown documents (`data/static/*.md`) → {chunk_count} chunks ingested.
- Dataset sizes: {retrieval_count} retrieval cases, {unanswerable_count} unanswerable questions, \
{routing_count} tool-routing cases (`parking_chatbot/evaluation/dataset.py`)."""

METRIC_DEFINITIONS = """\
These two metrics are **not** computed at the same level:

- **Precision@K** = of the K chunks retrieval returns, the fraction whose *source document* \
is one of the question's expected documents. The denominator is a count of **chunks** (K); \
a chunk counts as correct if its parent document is expected, not based on whether that \
specific chunk's text answers the question. The same document can supply more than one of \
the K chunks, and does count more than once here.
- **Recall@K** = of the question's expected *documents*, the fraction that are the source of \
at least one of the K retrieved chunks. The denominator is a count of **documents** (the size \
of the expected-source set), not chunks.

Because of this, Precision@K and Recall@K answer different questions and should not be read \
as two views of the same ratio: Precision@K needs most of the K slots to come from expected \
documents, while Recall@K only needs each expected document to show up at least once among \
those slots, however many slots it takes. A question whose answer lives in one small document, \
surrounded by four slots that correctly go to other relevant-but-unexpected documents, scores \
low Precision@K but can still score full Recall@K."""

# Frozen: this run used dataset labels that no longer exist in dataset.py (see
# LABEL_CORRECTIONS there), so it can never be reproduced and is recorded here verbatim
# instead of being recomputed. The "after" side is the live Retrieval quality table,
# spliced in by report.py right after this text.
BEFORE_CORRECTION_INTRO = """\
**Before / after.** Retrieval itself (the vector store, the chunking, the queries) did not \
change between these two runs — only the dataset's labels did, plus the one question that \
moved in from the unanswerable set. This isolates the effect of the correction from any change \
in retrieval behavior.

Before correction (29 cases, run 2026-11-01 22:43, using the labels now known to be wrong):

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | 0.66 | 0.60 |
| 3 | 0.60 | 0.95 |
| 5 | 0.46 | 0.98 |

1 of 29 cases had Recall@5 < 1 — the emergency-phones question, solely because its expected \
sources were wrong."""

LIMITATIONS_TEMPLATE = """\
- **Small corpus inflates Recall@{top_k}.** {doc_count} documents and only {chunk_count} chunks \
total means K={top_k} already covers a large slice of the whole corpus for many questions. High \
Recall@{top_k} here mainly shows the corpus is small enough that the right document is hard to \
miss, not that ranking is fine-grained; it may not hold on a larger document set.
- **Small sample sizes.** {retrieval_count} retrieval questions, {routing_count} tool-routing \
questions, {unanswerable_count} honesty questions ({honesty_judgments} judgments across \
{honesty_runs} runs). One question's result is worth several percentage points at this scale, \
so these scores are rough indicators, not precise measurements, and a couple of different or \
added questions can move a total noticeably.
- **p95 latency is computed from only {routing_count} samples** (the tool-routing / \
full-agent-turn set). A p95 over {routing_count} points is close to just "the 2nd-highest value \
measured," not a statistically stable 95th percentile; many more samples would be needed for that.
- **The LLM-as-judge for honesty is unvalidated.** No human-labelled ground truth was used to \
check the judge's own accuracy — the honesty score reflects the judge's criteria as much as \
the chatbot's behaviour. One judge-prompt bug (penalizing true supplementary facts as \
"invented") was already found and fixed during this evaluation, which shows the judge itself \
can be wrong.
- **The latency rows are not drawn from the same question set.** Retrieval and generation \
latency are measured over the {retrieval_count} retrieval questions; full-agent-turn latency is \
measured over the separate, smaller {routing_count}-question tool-routing set. Comparing the \
rows shows roughly how costly each stage is, not a sample-for-sample breakdown of the same calls.
- **Retrieval latency is mostly the embedding API call, not local vector search.** For each \
query, the retrieval time above is dominated by the network round trip to embed the query text; \
Chroma's local similarity search itself is fast. The retrieval row is closer to "embedding API \
latency" than "vector database performance.\""""

PRECISION_AT_1_EXPLANATION = (
    "for about a third of questions, the single best-matching chunk comes from a document "
    "other than the expected one — usually because a fact (for example EV charging) is "
    "mentioned in more than one document, or because a question's best single chunk doesn't "
    "fully cover a multi-document answer."
)

HONESTY_CAVEAT = (
    "Given the small sample size and the unvalidated judge (see Limitations), this is "
    "encouraging rather than conclusive."
)

LATENCY_EMBEDDING_NOTE = (
    "consistent with retrieval latency being mostly an embedding-API round trip rather than "
    "local search time."
)
