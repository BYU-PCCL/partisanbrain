from transformers import GPT2Tokenizer
import sys

# need to have pytorch and huggingface transformers installed to run this.
# can just run pip install transformers[torch] if you need both, or 
# just pip install transformers if you have torch

def cost_approximation(prompt, engine):
	possible_engines = ["davinci", "curie", "babbage", "ada"]
	assert engine in possible_engines, f"{engine} is not a valud engine"

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	num_tokens = len(tokenizer(prompt)['input_ids'])

	if engine == "davinci":
		cost = (num_tokens / 1000) * 0.0600
	elif engine == "curie":
		cost = (num_tokens / 1000) * 0.0060
	elif engine == "babbage":
		cost = (num_tokens / 1000) * 0.0012
	else:
		cost = (num_tokens / 1000) * 0.0008

	return cost

if __name__ == "__main__":
	# input prompt followed by engine when running the file
	# example: python3 cost_approximator.py "Hello World" curie
	# should print 1.2e-05
	cost = cost_approximation(sys.argv[1], sys.argv[2])
	print(cost)