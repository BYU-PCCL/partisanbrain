import sys
import pandas as pd
import pickle
from tqdm import tqdm
import os

if sys.argv[1] == '2012':
    from anes2012 import *
if sys.argv[1] == '2016':
    from anes2016 import *
if sys.argv[1] == '2020':
    from anes2020 import *

from common import *

#
# ============================================================================================
# ============================================================================================
#

foi_keys = fields_of_interest.keys()

def gen_backstory( pid, df ):
    person = df.iloc[pid]

    backstory = ""

    for k in foi_keys:
        anes_val = person[k]
        elem_template = fields_of_interest[k]['template']
        elem_map = fields_of_interest[k]['valmap']

        if len(elem_map) == 0:
            backstory += " " + elem_template.replace( 'XXX', str(anes_val) )

        elif anes_val in elem_map:
            backstory += " " + elem_template.replace( 'XXX', elem_map[anes_val] )

    if backstory[0] == ' ':
        backstory = backstory[1:]

    return backstory

#
# ============================================================================================
# ============================================================================================
#

anesdf = pd.read_csv( ANES_FN, sep=SEP, encoding='latin-1' )

foi_keys = list( fields_of_interest.keys() )
# for each field of interest, there is valmap and colname
for k in foi_keys:
    # for each key, make a new colunm
    colname, valmap = fields_of_interest[k]['colname'], fields_of_interest[k]['valmap']
    anesdf[colname] = anesdf[k].map( valmap )


# add new columns: prompt token_sets, ground_truth, and template_name
# template name is "first_person_backstory"
anesdf['template_name'] = "first_person_backstory"
# prompt is empty to start
anesdf['prompt'] = ""
# token sets is tok_sets
anesdf['token_sets'] = [tok_sets] * len(anesdf)
# ground truth is same column as VOTE_COL
anesdf['ground_truth'] = anesdf[VOTE_COL].map(VOTE_MAP)


full_results = []
prompts = []
for pid in tqdm( range(len(anesdf)) ):

    if "V200003" in anesdf.iloc[pid] and anesdf.iloc[pid]["V200003"]==2:
        print( f"SKIPPING {pid}..." )
        # we want to exclude cases marked as 2 on this variable; those are the panel respondents (interviewed in 2016 and 2020)
        prompts.append( "" )
        continue

    anes_id = anesdf.iloc[pid][ID_COL]

    prompt = gen_backstory( pid, anesdf )
    prompt += " " + query
    prompts.append( prompt )

    #print("---------------------------------------------------")
    #print( prompt )

    # set prompt to the generated prompt
    # anesdf.iloc[pid]['prompt'] = prompt
    # results = run_prompts( [prompt], tok_sets, engine="davinci" )
    #print(results[0][0])
    # full_results.append( (anes_id, prompt, results) )

anesdf['prompt'] = prompts
# drop anesdf where prompt is empty
anesdf = anesdf[anesdf['prompt'] != ""]

# check if ../data/anes{year} exists, if not, create it
if not os.path.exists( f"../data/anes{sys.argv[1]}" ):
    os.makedirs( f"../data/anes{sys.argv[1]}" )

# save to dir as ds.pkl
anesdf.to_pickle(f"../data/anes{sys.argv[1]}/ds.pkl")
pass