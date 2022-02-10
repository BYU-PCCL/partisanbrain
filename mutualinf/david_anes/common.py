import openai
import numpy as np
import time

openai.api_key = "sk-haGATYseQkMnUMooCHP4YvEFDGHnAzY4GJiJeUtv"

#Publishable pk-VJvX6FYFtbjuC8PqgFByb4f7cBORFQQ13bup1Nuk

def lc( t ):
    return t.lower()

def uc( t ):
    return t.upper()

def mc( t ):
    tmp = t.lower()
    return tmp[0].upper() + t[1:]

def gen_variants( toks ):
    results = []

    variants = [ lc, uc, mc ]

    for t in toks:
        for v in variants:
            results.append( " " + v(t) )

    return results

def logsumexp( log_probs ):
    log_probs = log_probs - np.max(log_probs)
    log_probs = np.exp(log_probs)
    log_probs = log_probs / np.sum( log_probs )
    return log_probs

def extract_probs( lp ):
    lp_keys = list( lp.keys() )
    ps = [ lp[k] for k in lp_keys ]
    ps = logsumexp( np.asarray(ps) )
    vals = [ (lp_keys[ind], ps[ind]) for ind in range(len(lp_keys)) ]

    vals = sorted( vals, key=lambda x: x[1], reverse=True )

    result = {}
    for v in vals:
        result[ v[0] ] = v[1]

    return result

def do_query( prompt, max_tokens=2, engine="davinci" ):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )

    token_responses = response['choices'][0]['logprobs']['top_logprobs']

    results = []
    for ind in range(len(token_responses)):
        results.append( extract_probs( token_responses[ind] ) )

    return results, response

def collapse_r( response, toks ):
    total_prob = 0.0
    for t in toks:
        if t in response:
            total_prob += response[t]
    return total_prob

def print_response( template_val, tok_sets, response ):
    #print( f"{template_val}" )

    print( tok_sets )

    tr = []
    for tok_set_key in tok_sets.keys():
        toks = tok_sets[tok_set_key]
        full_prob = collapse_r( response[0], toks )
        tr.append( full_prob )
        #print( f";{tok_set_key};{full_prob}", end="" )
        #print( "\t{:.2f}".format(full_prob), end="" )
    print("\t\t",end="")
    tr = np.asarray( tr )
    tr = tr / np.sum(tr)
    for ind, tok_set_key in enumerate( tok_sets.keys() ):
        print( f"\t{tok_set_key}\t{tr[ind]}", end="" )
        #print( "\t{:.2f}".format(tr[ind]), end="" )
    print("")

def parse_response( template_val, tok_sets, response ):
    tr = []
    for tok_set_key in tok_sets.keys():
        toks = tok_sets[tok_set_key]
        full_prob = collapse_r( response[0], toks )
        tr.append( full_prob )
    tr = np.asarray( tr )
    tr = tr / np.sum(tr)

    results = {}
    for ind, tok_set_key in enumerate( tok_sets.keys() ):
        results[ tok_set_key ] = tr[ind]
    return results


def run_prompts( prompts, tok_sets, engine="davinci" ):
    results = []
    for prompt in prompts:
        #print("---------------------------------------------------")
        #print( prompt )
        response, full_response = do_query( prompt, max_tokens = 2, engine=engine )
        #print( response )
        #print_response( prompt, tok_sets, response )
        simp_results = parse_response( prompt, tok_sets, response )
        #print( simp_results )
        time.sleep( 0.1 )
        results.append( (simp_results, response, full_response) )
    return results

def run_experiment( template, template_vals, tok_sets ):
    prompts = []
    for template_val in template_vals:
        grounded_prompt = template.replace( "XXX", template_val )
        prompts.append( grounded_prompt )
    return run_prompts( prompts, tok_sets )
