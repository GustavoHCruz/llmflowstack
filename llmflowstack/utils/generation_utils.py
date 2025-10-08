from transformers.generation.configuration_utils import GenerationConfig

from llmflowstack.schemas.params import GenerationParams


def create_generation_params(generation_configs: GenerationParams) -> GenerationConfig:

	params = {
		"max_new_tokens": generation_configs.max_new_tokens,
		"repetition_penalty": generation_configs.repetition_penalty
	}
	if generation_configs.sample:
		sample = generation_configs.sample
		params.update({
			"do_sample": True,
			"temperature": sample.temperature,
			"top_p": sample.top_p,
			"typical_p": sample.typical_p,
			"num_beams": 1
		})
	elif generation_configs.beams == "beams":
		beams = generation_configs.beams
		params.update({
			"do_sample": False,
			"num_beams": beams.num_beams,
			"length_penalty": beams.length_penalty,
			"early_stopping": beams.early_stopping
		})

	return GenerationConfig(**params)