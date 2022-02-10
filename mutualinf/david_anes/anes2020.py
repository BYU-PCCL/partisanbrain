from common import *
from anes_common import *

SEP=','
OUTPUT_FN = "./full_results_2020.pkl"
OUTPUT_CSV = "./full_results_2020.csv"
ID_COL = "V200001"
VOTE_COL = "V202110x" # 1-Biden, 2-Trump, 3-Jorgensen, 4-Hawkins
VOTE_MAP = {
    1: "biden",
    2: "trump",
    3: "jorgensen",
    4: "hawkins",
}
K1 = "trump"
K2 = "biden"

tok_sets = {
    "trump": gen_variants( [ "donald", "trump", "republican", "conservative" ] ),
    "biden": gen_variants( [ "joe", "joseph", "biden", "democrat", "democratic", "liberal"] ),
    }

query = "In the 2020 presidential election, Donald Trump is the Republican candidate, and Joe Biden is the Democratic candidate, and I voted for"

ANES_FN = './2020 ANES.csv'
fields_of_interest = {
    # race  1= white 2= black 3 = asian 5 = hispanic
    'V201549x': {
        "template":"Racially, I am XXX.",
        "valmap":{ 1:'white', 2:'black', 3:'asian', 4:'native American', 5:'hispanic' },
        "colname": "race",
        },

    # discuss_politics 1=yes discuss politics, 2=never discuss politics
    'V202022': {
        "template":"XXX",
        "valmap":{1:'I like to discuss politics with my family and friends.', 2:'I never discuss politics with my family or friends.'},
        "colname": "discuss_politics",
        },

    # ideology 1-7 = extremely liberal, liberal, slightly liberal, moderate, slightly conservative, conservative, extremely conservative
    'V201200': {
        "template":"Ideologically, I am XXX.",
        "valmap":{1:"extremely liberal",2:"liberal",3:"slightly liberal",4:"moderate",5:"slightly conservative",6:"conservative",7:"extremely conservative"},
        "colname": "ideology",
        },

    # party
    'V201231x': {
        "template":"Politically, I am XXX.",
        "valmap":{1:"a strong democrat", 2:"a weak Democrat", 3:"an independent who leans Democratic", 4:"an independent",5:"an independent who leans Republican", 6:"a weak Republican",7:"a strong Republican"},
        "colname": "party",
        },

    # church_goer
    'V201452': {
        "template":"I XXX.",
        "valmap":{ 1:"attend church", 2:"do not attend church"},
        "colname": "church_goer",
        },

    # age
    'V201507x': {
        "template":"I am XXX years old.",
        "valmap":{},
        "colname": "age",
        },

    # gender 1=male  2=female
    'V201600': {
        "template":"I am a XXX.",
        "valmap":{ 1:"man", 2:"woman"},
        "colname": "gender"
        },

    # political_interest = if_else(V162256 > 0, V162256, NA_real_),
    'V202406': {
        "template":"I am XXX interested in politics.",
        "valmap":{1:"very", 2:"somewhat", 3:"not very", 4:"not at all"},
        "colname": "political_interest",
        },

    # NOT ASKED IN 2020
    # patriotism = if_else(V162125x > 0, V162125x, NA_real_))
#    '': {
#        "template":"It makes me feel XXX to see the American flag.",
#        "valmap":{1:"extremely good", 2:"moderately good", 3:"a little good", 4:"neither good nor bad", 5:"a little bad", 6:"moderately bad", 7:"extremely bad"},
#        },

    # this is sample address, which may be different than registration address...?
    'V201007d': {
        "template":"I am from XXX.",
        "valmap":fips_state_map,
        "colname": "state",
        },
    

    }
