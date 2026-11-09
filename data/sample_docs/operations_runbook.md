# Enterprise Search Operations Runbook

## Latency triage

When answer latency rises, first separate retrieval time from reranking and generation time. Check whether dense vector search, BM25 lookup, or the cross-encoder reranker became slower after a deployment or index refresh. A sudden increase in candidate count can raise reranking cost even when the vector index is healthy.

Next, inspect document refresh jobs for unusually large files, repeated retries, and embedding backlogs. Timeouts should be recorded by stage so an operator can tell whether the bottleneck is ingestion, retrieval, reranking, or the API layer. Do not hide a failing stage by silently returning unranked results; surface the degraded mode in service health and logs.

## Quality checks

Review a small set of representative questions after each index or model change. Check that the first cited source actually supports the answer, that exact identifiers are found by keyword search, and that paraphrased questions are found by dense retrieval. Faithfulness checks should flag any answer sentence that lacks a supporting citation.

## Refresh procedure

Load changed documents, rebuild their chunks, and generate embeddings before replacing the previous index version. Keep the prior index available until smoke tests pass. A failed refresh should leave the last known good index in service.
