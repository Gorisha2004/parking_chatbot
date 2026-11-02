# Stage 1 Evaluation Report

Generated: 2026-11-02 12:01

## Setup

- LLM: `gpt-4.1-mini-2025-04-14` (DIAL / Azure OpenAI-compatible), temperature 0.
- Embedding model: `text-embedding-3-small-1`.
- Chunking: 300 characters per chunk, 50 characters overlap (`RecursiveCharacterTextSplitter`).
- Vector database: Chroma, collection `parking_static_docs`, persisted locally.
- Retrieval: top-K = 5 (`config.TOP_K`) — the same K the live chatbot uses for `search_parking_info`.
- Corpus: 9 static markdown documents (`data/static/*.md`) → 50 chunks ingested.
- Dataset sizes: 30 retrieval cases, 7 unanswerable questions, 18 tool-routing cases (`parking_chatbot/evaluation/dataset.py`).

## Method

- Retrieval quality: Precision@K / Recall@K computed against a hand-labelled dataset of 30 questions (`parking_chatbot/evaluation/dataset.py`), each with the static-doc source(s) it should be retrieved from. See "Precision@K and Recall@K, precisely" below — the two metrics are computed at different levels.
- Generation latency: the RAG chain's `generate_answer` is timed separately from retrieval, given already-retrieved context, over the same question set.
- Tool-routing accuracy and agent latency: 18 questions sent through the full agent (`chatbot.agent.chat`), checking which tool(s) it called.
- Honesty on unanswerable questions: 7 questions with no answer in the documentation, graded HONEST/INVENTED by an LLM-as-judge. Because LLM answers vary between runs, the set is asked 3 times and the range is reported.
- Latency is end-to-end wall-clock time per call (`time.perf_counter()`); mean, p95 and max are reported. Reproduce with `uv run python scripts/run_evaluation.py`.

### Precision@K and Recall@K, precisely

These two metrics are **not** computed at the same level:

- **Precision@K** = of the K chunks retrieval returns, the fraction whose *source document* is one of the question's expected documents. The denominator is a count of **chunks** (K); a chunk counts as correct if its parent document is expected, not based on whether that specific chunk's text answers the question. The same document can supply more than one of the K chunks, and does count more than once here.
- **Recall@K** = of the question's expected *documents*, the fraction that are the source of at least one of the K retrieved chunks. The denominator is a count of **documents** (the size of the expected-source set), not chunks.

Because of this, Precision@K and Recall@K answer different questions and should not be read as two views of the same ratio: Precision@K needs most of the K slots to come from expected documents, while Recall@K only needs each expected document to show up at least once among those slots, however many slots it takes. A question whose answer lives in one small document, surrounded by four slots that correctly go to other relevant-but-unexpected documents, scores low Precision@K but can still score full Recall@K.

## Dataset corrections

Labels changed after a run showed them contradicting the source documents (each checked against `data/static/*.md`):

- Emergency phones question: expected sources changed from contact_info + facilities to contact_info only; the emergency phones are described in contact_info.md, not in facilities.md.
- Office / online question: moved from the unanswerable set to the retrieval set (expected contact_info); contact_info.md gives the administration office address, so the documents do answer it.

**Before / after.** Retrieval itself (the vector store, the chunking, the queries) did not change between these two runs — only the dataset's labels did, plus the one question that moved in from the unanswerable set. This isolates the effect of the correction from any change in retrieval behavior.

Before correction (29 cases, run 2026-11-01 22:43, using the labels now known to be wrong):

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | 0.66 | 0.60 |
| 3 | 0.60 | 0.95 |
| 5 | 0.46 | 0.98 |

1 of 29 cases had Recall@5 < 1 — the emergency-phones question, solely because its expected sources were wrong.

After correction (30 cases, current run):

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | 0.67 | 0.63 |
| 3 | 0.59 | 0.97 |
| 5 | 0.45 | 1.00 |

Every case achieved full Recall@5.

## Retrieval quality (Precision@K / Recall@K)

| K | Precision@K | Recall@K |
|---|---|---|
| 1 | 0.67 | 0.63 |
| 3 | 0.59 | 0.97 |
| 5 | 0.45 | 1.00 |

Every case achieved full Recall@5.

## Latency (seconds)

| Stage | Mean | p95 | Max |
|---|---|---|---|
| Retrieval only | 0.58s | 0.93s | 1.44s |
| Generation only (RAG chain) | 2.28s | 3.18s | 3.74s |
| Full agent turn (incl. tool calls) | 3.70s | 6.20s | 6.52s |

## Tool-routing accuracy

18/18 = 1.00

## Honesty on unanswerable questions

Mean 1.00 answered honestly (refused or pointed to support instead of inventing an answer); per run: 1.00, 1.00, 1.00 (range 1.00–1.00).

| Question | Honest runs | Example reply (an invented one if any run invented) |
|---|---|---|
| Do you offer a discount for a yearly subscription? | 3/3 | I don't have information about discounts for yearly subscriptions at CityPark Central. For details on such offers, please contact the customer support of the parking facility. |
| Is every level assigned for one kind of vehicle parking? | 3/3 | The information I have does not specify that each level at CityPark Central is assigned to a specific kind of vehicle parking. The facility has 4 levels and allows cars, SUVs (including electric cars), motorcycles, scooters, two-wheelers, and three-wheelers such as auto-rickshaws to park. Electric … |
| What are the offers present if any customer wants to buy a subscription for the complete year? | 3/3 | I don't have information about yearly subscriptions or special offers for them at CityPark Central. If you want, I can help you with current prices for hourly, daily, or weekly parking. For subscription or offer details, please contact CityPark Central's customer support. |
| Is your parking area registered and authorized to park vehicles? | 3/3 | The parking facility is equipped with CCTV, security patrols, and lighting to ensure safety. While it is secure, the facility is not responsible for valuables left inside vehicles. Overnight parking is allowed only with a confirmed reservation. If you need more specific information about registrati… |
| Is WiFi available in the parking facility? | 3/3 | The information I have does not mention WiFi availability in the parking facility. For details about WiFi, I suggest contacting CityPark Central's customer support directly. Is there anything else I can help you with? |
| Can I reserve a specific parking level or spot? | 3/3 | At CityPark Central, you can reserve a parking space for your vehicle, but the information I have does not mention the option to reserve a specific parking level or a specific parking spot. If you want, I can help you with a reservation for your vehicle without specifying the level or spot. Would y… |
| What is your refund policy for damaged vehicles? | 3/3 | I don't have information about the refund policy for damaged vehicles. For questions about liability or refunds related to vehicle damage, please contact CityPark Central's customer support directly. If you need help with anything else, just ask! |

## Limitations

- **Small corpus inflates Recall@5.** 9 documents and only 50 chunks total means K=5 already covers a large slice of the whole corpus for many questions. High Recall@5 here mainly shows the corpus is small enough that the right document is hard to miss, not that ranking is fine-grained; it may not hold on a larger document set.
- **Small sample sizes.** 30 retrieval questions, 18 tool-routing questions, 7 honesty questions (21 judgments across 3 runs). One question's result is worth several percentage points at this scale, so these scores are rough indicators, not precise measurements, and a couple of different or added questions can move a total noticeably.
- **p95 latency is computed from only 18 samples** (the tool-routing / full-agent-turn set). A p95 over 18 points is close to just "the 2nd-highest value measured," not a statistically stable 95th percentile; many more samples would be needed for that.
- **The LLM-as-judge for honesty is unvalidated.** No human-labelled ground truth was used to check the judge's own accuracy — the honesty score reflects the judge's criteria as much as the chatbot's behaviour. One judge-prompt bug (penalizing true supplementary facts as "invented") was already found and fixed during this evaluation, which shows the judge itself can be wrong.
- **The latency rows are not drawn from the same question set.** Retrieval and generation latency are measured over the 30 retrieval questions; full-agent-turn latency is measured over the separate, smaller 18-question tool-routing set. Comparing the rows shows roughly how costly each stage is, not a sample-for-sample breakdown of the same calls.
- **Retrieval latency is mostly the embedding API call, not local vector search.** For each query, the retrieval time above is dominated by the network round trip to embed the query text; Chroma's local similarity search itself is fast. The retrieval row is closer to "embedding API latency" than "vector database performance."

## Conclusions

- Retrieval finds the right document almost every time (Recall@5 = 1.00). The weak point is Precision@1 = 0.67: for about a third of questions, the single best-matching chunk comes from a document other than the expected one — usually because a fact (for example EV charging) is mentioned in more than one document, or because a question's best single chunk doesn't fully cover a multi-document answer.
- Tool routing is reliable on this test set (100% correct over 18 questions).
- The agent avoided inventing answers for out-of-scope questions throughout this run (100% honest across 3 repeats of 7 questions). Given the small sample size and the unvalidated judge (see Limitations), this is encouraging rather than conclusive.
- Generation accounts for a larger share of end-to-end latency than retrieval, consistent with retrieval latency being mostly an embedding-API round trip rather than local search time.
