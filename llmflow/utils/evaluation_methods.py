from evaluate import load
from nltk.stem.snowball import SnowballStemmer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def stem_texts(texts: list[str]) -> list[str]:
	stemmer = SnowballStemmer("portuguese")

	stemmed_texts: list[str] = []
	for text in texts:
		stemmed_text = " ".join([stemmer.stem(word) for word in text.split()])
		stemmed_texts.append(stemmed_text)

	return stemmed_texts

def rouge_evaluation(
	preds: list[str],
	refs: list[str]
) -> dict[str, float]:
	preds_stemmed = stem_texts(preds)
	refs_stemmed = stem_texts(refs)

	rouge_metrics = {"rouge1": [], "rouge2": [], "rougeL": []}
	scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

	for ref, pred in zip(refs_stemmed, preds_stemmed):
		scores = scorer.score(ref, pred)
		for key in rouge_metrics:
			rouge_metrics[key].append(scores[key].fmeasure)

	return {k: sum(v)/len(v) for k, v in rouge_metrics.items()}

def bert_score_evaluation(
	preds: list[str],
	refs: list[str]
) -> dict[str, float]:
	bertscore = load("bertscore")

	bert_result = bertscore.compute(predictions=preds, references=refs, lang="pt")

	bert_avg = {}
	if bert_result:
		bert_avg = {
			"bertscore_precision": sum(bert_result["precision"]) / len(bert_result["precision"]),
			"bertscore_recall": sum(bert_result["recall"]) / len(bert_result["recall"]),
			"bertscore_f1": sum(bert_result["f1"]) / len(bert_result["f1"])
		}

	return bert_avg

def cosine_similarity_evaluation(
	preds: list[str],
	refs: list[str]
) -> dict[str, float]:
	model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

	emb_preds = model.encode(preds, convert_to_tensor=True)
	emb_refs = model.encode(refs, convert_to_tensor=True)

	cos_sim_matrix = util.cos_sim(emb_preds, emb_refs)

	cos_sim_scores = cos_sim_matrix.diag()  
	avg_cos_sim = cos_sim_scores.mean().item()

	return {"cosine_similarity": float(avg_cos_sim)}

def text_evaluation(
	preds: list[str],
	refs: list[str],
	rouge: bool = True,
	bert: bool = True,
	cosine: bool = True
) -> dict[str, float]:
	result = {}
	if rouge:
		result.update(rouge_evaluation(
			preds=preds,
			refs=refs
		))
	if bert:
		result.update(bert_score_evaluation(
			preds=preds,
			refs=refs
		))
	if cosine:
		result.update(cosine_similarity_evaluation(
			preds=preds,
			refs=refs
		))

	return result