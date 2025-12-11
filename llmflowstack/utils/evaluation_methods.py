from typing import Literal

from evaluate import load
from nltk.stem.snowball import SnowballStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def avg(
	values: list[float] | None
) -> float:
	return sum(values) / len(values) if values else 0.0

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
) -> dict[Literal["rouge1", "rouge2", "rougeL"], float]:
	preds_stemmed = stem_texts(preds)
	refs_stemmed = stem_texts(refs)

	rouge_metrics = {"rouge1": [], "rouge2": [], "rougeL": []}
	scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

	for ref, pred in zip(refs_stemmed, preds_stemmed):
		scores = scorer.score(
			target=ref,
			prediction=pred
		)
		for key in rouge_metrics:
			rouge_metrics[key].append(scores[key].fmeasure)

	rouge1 = round(avg(rouge_metrics["rouge1"]), 4)
	rouge2 = round(avg(rouge_metrics["rouge2"]), 4)
	rougeL = round(avg(rouge_metrics["rougeL"]), 4)

	return {
		"rouge1": rouge1,
		"rouge2": rouge2,
		"rougeL": rougeL
	}

def bert_score_evaluation(
	preds: list[str],
	refs: list[str],
	encoder: str | None = None,
	lang: str = "pt"
) -> dict[Literal["bertscore_precision", "bertscore_recall", "bertscore_f1"], float]:
	bertscore = load("bertscore")

	bert_score = bertscore.compute(
		predictions=preds,
		references=refs,
		model_type=encoder,
		lang=lang
	)

	assert bert_score is not None

	precision = round(avg(bert_score["precision"]), 4)
	recall = round(avg(bert_score["recall"]), 4)
	f1 = round(avg(bert_score["f1"]), 4)

	return {
		"bertscore_precision": precision,
		"bertscore_recall": recall,
		"bertscore_f1": f1
	}

def cosine_similarity_evaluation(
	preds: list[str],
	refs: list[str],
	encoder: str | None = None
) -> dict[Literal["cosine_similarity"], float]:
	if not encoder:
		encoder = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
	
	model = SentenceTransformer(
		encoder,
		trust_remote_code=True
	)

	try:
		emb_preds = model.encode(preds, task="retrieval", convert_to_tensor=True)	
		emb_refs = model.encode(refs, task="retrieval", convert_to_tensor=True)
	except TypeError:
		emb_preds = model.encode(preds, convert_to_tensor=True)	
		emb_refs = model.encode(refs, convert_to_tensor=True)

	cos_sim_matrix = util.cos_sim(emb_preds, emb_refs)

	cos_sim_scores = cos_sim_matrix.diag()  
	avg_cos_sim = round(float(cos_sim_scores.mean().item()), 4)

	return {"cosine_similarity": float(avg_cos_sim)}

def bleu_score_evaluation(
	preds: list[str],
	refs: list[str]
) -> dict[Literal["bleu_score"], float]:
	smooth = SmoothingFunction().method1

	scores = []
	for pred, ref in zip(preds, refs):
		if not pred.strip() or not ref.strip():
			scores.append(0.0)
			continue
		scores.append(sentence_bleu(
			references=[ref.split()],
			hypothesis=pred.split(),
			smoothing_function=smooth
		))

	bleu_score = round(avg(scores), 4)

	return {
		"bleu_score": bleu_score
	}

def text_evaluation(
	preds: list[str],
	refs: list[str],
	rouge: bool = True,
	bert: bool = True,
	cosine: bool = True,
	bleu: bool = True,
	encoder: str | None = None,
	lang: str = "pt"
) -> dict[str, float]:
	result = {}
	if bert:
		result.update(bert_score_evaluation(
			preds=preds,
			refs=refs,
			encoder=encoder,
			lang=lang
		))
	if bleu:
		result.update(bleu_score_evaluation(
			preds=preds,
			refs=refs
		))
	if cosine:
		result.update(cosine_similarity_evaluation(
			preds=preds,
			refs=refs,
			encoder=encoder
		))
	if rouge:
		result.update(rouge_evaluation(
			preds=preds,
			refs=refs
		))

	return result