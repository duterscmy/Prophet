"""Local, deterministic sentence-level ROUGE metrics for XSum-500."""

from rouge_score import rouge_scorer


_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def process_results(doc, results):
    prediction = str(results[0]).strip()
    reference = str(doc["summary"]).strip()
    scores = _SCORER.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
