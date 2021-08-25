import sys
sys.path.append('src')
from tqdm import tqdm
import os
import pickle
import pandas as pd
from pdb import set_trace as bp

from common import *

#
# ==========================================================================
#


data_dir = "data/lucid"

exp_dir = "experiments/lucid"

def fixstr( s ):
    s = str( s )
    if s == 'nan':
        return ''
    return s

pp_df = pd.read_csv( os.path.join(data_dir, "Pigeonholing full file for Chris_updated.csv" ))

human_d_words = {}
human_r_words = {}
dmap = {}
for row in pp_df.iterrows():
    id = row[1]['ID']
    dmap[id] = row[1]
    human_d_words[id] = []
    human_d_words[id].append( fixstr(row[1]['GenD_1']) )
    human_d_words[id].append( fixstr(row[1]['GenD_2']) )
    human_d_words[id].append( fixstr(row[1]['GenD_3']) )
    human_d_words[id].append( fixstr(row[1]['GenD_4']) )

    human_r_words[id] = []
    human_r_words[id].append( fixstr(row[1]['GenR_1']) )
    human_r_words[id].append( fixstr(row[1]['GenR_2']) )
    human_r_words[id].append( fixstr(row[1]['GenR_3']) )
    human_r_words[id].append( fixstr(row[1]['GenR_4']) )

gpt3_df = pd.read_csv( os.path.join(data_dir, "gpt3_uber_final.csv" ))

gpt3_d_words = {}
gpt3_r_words = {}
for row in gpt3_df.iterrows():
    id = row[1]['ID']
    gpt3_d_words[id] = []
    gpt3_d_words[id].append( fixstr(row[1]['GD1']) )
    gpt3_d_words[id].append( fixstr(row[1]['GD2']) )
    gpt3_d_words[id].append( fixstr(row[1]['GD3']) )
    gpt3_d_words[id].append( fixstr(row[1]['GD4']) )

    gpt3_r_words[id] = []
    gpt3_r_words[id].append( fixstr(row[1]['GR1']) )
    gpt3_r_words[id].append( fixstr(row[1]['GR2']) )
    gpt3_r_words[id].append( fixstr(row[1]['GR3']) )
    gpt3_r_words[id].append( fixstr(row[1]['GR4']) )

#
# ==========================================================================
#

def replace_words( template, word_list ):
    grounded_prompt = template

    # word_list.append( [''] * (4-len(word_list)) )

    grounded_prompt = grounded_prompt.replace( "AAA", word_list[0] )
    grounded_prompt = grounded_prompt.replace( "BBB", word_list[1] )
    grounded_prompt = grounded_prompt.replace( "CCC", word_list[2] )
    grounded_prompt = grounded_prompt.replace( "DDD", word_list[3] )

    return grounded_prompt

#
# ==========================================================================
#

def proc_word_set( word_set, template, tok_sets, dmap ):
    results = {}
    for id in tqdm( word_set.keys() ):
        try:
            print( "================================================================" )
            print( "" )

            grounded_prompt = replace_words( template, word_set[id] )
            print( grounded_prompt )

            response, full_response = do_query( grounded_prompt, max_tokens=1 )
            probs = collapse_probs( tok_sets, response )
            results[id] = [ probs, full_response ]

#           print( full_response )
            print( f"True background is: {dmap[id]['Ideo']}" )
            print_response( tok_sets, probs )

        except:
            print("Some sort of error!")
            pass

    return results

#
# ==========================================================================
#


template = """PREFIX
1. AAA 2. BBB 3. CCC 4. DDD :"""

qas = [
    {
        "tag":"positivity",
        "prefix":"Are the following descriptions of PARTY positive or negative?\n\n1. Friendly 2. Happy 3. Funny 4. Kind : positive\n1. Angry 2. Mean 3. Disgusting 4. Awful : negative",
        "postfix":"",
        "tok_set": {
            "positive": gen_variants( [ "positive" ] ),
            "negative": gen_variants( [ "negative" ] ),
        }
    },

    {
        "tag":"extremity",
        "prefix":"Are the following descriptions of PARTY extreme?\n\n1. Terrorists 2. Evil 3. Disgusting 4. Animals : yes\n1. Friendly 2. Normal 3. Sensible 4. Informed : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },

    {
        "tag":"PID",
        "prefixd":"The following text was written about PARTY. Identify the political party (Democrat, Republican, Independent) of the person who wrote this text:\n\n1. Naive 2. Crazy 3. Entitled 4. Woke : Republican\n1. Considerate 2. Educated 3. Empathetic 4. Open-minded : Democrat",
        "prefixr":"The following text was written about PARTY. Identify the political party (Democrat, Republican, Independent) of the person who wrote this text:\n\n1. Faithful 2. Patriotic 3. Realistic 4. White : Republican\n1. Selfish 2. White 3. Rich 4. Ignorant : Democrat",
        "postfix":"",
        "tok_set": {
            "D": gen_variants( [ "democrat"  ] ),
            "R": gen_variants( [ "republican" ] ),
        }
    },


    {
        "tag":"groups",
        "prefix":"Do the following descriptions of PARTY mention social groups?\n\n1. Black 2. Socialist 3. Atheists 4. Elites : yes\n1. Abortion 2. Welfare 3. Entitled 4. Free money : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },


    {
        "tag":"traits",
        "prefix":"Do the following descriptions of PARTY mention personality or character traits?\n\n1. Kind 2. Caring 3. Upstanding 4. Honest : yes\n1. Abortion 2. Welfare 3. Taxes 4. Corruption : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },


    {
        "tag":"issues",
        
        "prefix":"Do the following descriptions of PARTY include government or policy issues?\n\n1. Kind 2. Caring 3. Upstanding 4. Honest : no\n1. Abortion 2. Welfare 3. Taxes 4. Corruption : yes",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },

]
qas = [
    {
        "tag":"extremity1",
        "prefix":"Are the following descriptions of PARTY extreme?\n\n1. Terrorists 2. Evil 3. Disgusting 4. Animals : yes\n1. Friendly 2. Normal 3. Sensible 4. Informed : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },
    {
        "tag":"extremity2",
        "prefix":"Are the following descriptions of PARTY extreme?\n\n1. Terrorists 2. Evil 3. Disgusting 4. Animals : yes\n1. Friendly 2. Normal 3. Sensible 4. Informed : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },
    {
        "tag":"extremity3",
        "prefix":"Are the following descriptions of PARTY extreme?\n\n1. Terrorists 2. Evil 3. Disgusting 4. Animals : yes\n1. Friendly 2. Normal 3. Sensible 4. Informed : no",
        "postfix":"",
        "tok_set": {
            "yes": gen_variants( [ "yes" ] ),
            "no": gen_variants( [ "no" ] ),
        }
    },

]

parties = ["Republicans","Democrats"]

# capitalist, radical, socialist, feminist, classical, communist, leftist, green, fascist, modern, neo, pragmatic, marxist, libertarian
final_results = {}

for qa in qas:
    if not qa['tag'] in final_results:
        final_results[qa['tag']] = {}

    for party in parties:

        if not party in final_results[qa['tag']]:
            final_results[qa['tag']][party] = {}

        print( "================================================================" )
        print( f"{qa['tag']} - {party}" )

        tmp_template = template
        if qa['tag'] == "PID":
            if party == "Republicans":
                tmp_template = tmp_template.replace( "PREFIX", qa['prefixr'] )
            else:
                tmp_template = tmp_template.replace( "PREFIX", qa['prefixd'] )
        else:
            tmp_template = tmp_template.replace( "PREFIX", qa['prefix'] )
        tmp_template = tmp_template.replace( "PARTY", party )
        tmp_template = tmp_template.replace( "POSTFIX", qa['postfix'] )

        if party == "Republicans":
            human_logits = proc_word_set( human_r_words, tmp_template, qa['tok_set'], dmap )
            gpt3_logits = proc_word_set( gpt3_r_words, tmp_template, qa['tok_set'], dmap )

        else:
            human_logits = proc_word_set( human_d_words, tmp_template, qa['tok_set'], dmap )
            gpt3_logits = proc_word_set( gpt3_d_words, tmp_template, qa['tok_set'], dmap )

        final_results[qa['tag']][party]['human'] = human_logits
        final_results[qa['tag']][party]['gpt3'] = gpt3_logits

        with open(os.path.join(exp_dir,"progress.txt"), "a+") as f:
            f.write( f"{qa['tag']} - {party} finished\n" )

pickle.dump( final_results, open(os.path.join(exp_dir,"final_results.pkl"),"wb"))
bp()

#
# ==========================================================================
#

rmap = {}
for row in pp_df.iterrows():
    ideo = row[1]['Ideo']
    id = row[1]['ID']
    if not ideo in rmap:
        rmap[ideo] = []
    try:
#        rmap[ideo].append( final_results['positivity_33']['Republicans']['human'][id][0] )
#        rmap[ideo].append( final_results['positivity_33']['Republicans']['gpt3'][id][0] )
#        rmap[ideo].append( final_results['positivity_33']['Democrats']['human'][id][0] )
        rmap[ideo].append( final_results['positivity_33']['Democrats']['gpt3'][id][0]
        )
    except:
        print("Error!")
for k in rmap.keys():
    print( k, " ", np.mean( np.vstack( rmap[k] ), axis=0 ) )