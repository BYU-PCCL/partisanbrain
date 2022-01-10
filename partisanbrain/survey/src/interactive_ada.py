import heapq
import numpy as np
import openai
import os

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


def ada(prompt, k):
    print(prompt)
    response = openai.Completion.create(engine="davinci",
                                        prompt=prompt,
                                        max_tokens=1,
                                        logprobs=100)
    logits = response["choices"][0]["logprobs"]["top_logprobs"][0]
    logits_exp = {k: np.exp(v) for (k, v) in logits.items()}
    logits_exp_sum = sum(logits_exp.values())
    probs = {k: v / logits_exp_sum for (k, v) in logits_exp.items()}
    top_keys = heapq.nlargest(k, probs, key=probs.get)
    top_probs = {k: probs[k] for k in top_keys}
    return top_probs


if __name__ == '__main__':
    import sys
    prompt = " ".join(sys.argv[1:])
    probs = ada(prompt, 20)
    print(probs)
