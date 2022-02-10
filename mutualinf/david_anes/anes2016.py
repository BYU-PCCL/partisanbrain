from common import *
from anes_common import *

OUTPUT_FN = "./full_results_2016.pkl"
OUTPUT_CSV = "./full_results_2016.csv"
SEP='|'
ID_COL = "V160001_orig"
VOTE_COL = "V162062x"
VOTE_MAP = {
    1: "clinton",
    2: "trump",
}
K1 = "trump"
K2 = "clinton"

tok_sets = {
    "trump": gen_variants( [ "trump", "donald", "republican", "conservative" ] ),  # the republican, mr trump
    "clinton": gen_variants( [ "clinton", "hillary", "rodham", "senator", "democrat", "democratic", "liberal"] ),
    }

query = "In the 2016 presidential election, I voted for"

ANES_FN = '2016 ANES.csv'

fields_of_interest = {
    # race V161310x 1= white 2= black 3 = asian 5 = hispanic
    'V161310x': {
        "template":"Racially, I am XXX.",
        "valmap":{ 1:'white', 2:'black', 3:'asian', 4:'native American', 5:'hispanic' },
        "colname": "race",
        },

    # discuss_politics V162174 1=yes discuss politics, 2=never discuss politics
    'V162174': {
        "template":"XXX",
        "valmap":{1:'I like to discuss politics with my family and friends.', 2:'I never discuss politics with my family or friends.'},
        "colname": "discuss_politics",
        },

    # ideology V161126 1-7 = extremely liberal, liberal, slightly liberal, moderate, slightly conservative, conservative, extremely conservative
    'V161126': {
        "template":"Ideologically, I am XXX.",
        "valmap":{1:"extremely liberal",2:"liberal",3:"slightly liberal",4:"moderate",5:"slightly conservative",6:"conservative",7:"extremely conservative"},
        "colname": "ideology",
        },

    # party V161158x
    'V161158x': {
        "template":"Politically, I am XXX.",
        "valmap":{1:"a strong democrat", 2:"a weak Democrat", 3:"an independent who leans Democratic", 4:"an independent",5:"an independent who leans Republican", 6:"a weak Republican",7:"a strong Republican"},
        "colname": "party",
        },

    # church_goer V161244
    'V161244': {
        "template":"I XXX.",
        "valmap":{ 1:"attend church", 2:"do not attend church"},
        "colname": "church_goer",
        },

    # age V161267
    'V161267': {
        "template":"I am XXX years old.",
        "valmap":{},
        "colname": "age",
        },

    # education V161270 <= 9, "High School or Less", >= 10 & V161270 <= 12, "Some College / AA", 13, "Bachelor's", >=14 & V161270 <= 16, "Graduate Degree"
#    '': {
#        "template":"",
#        "valmap":{},
#        },

    # gender  V161342 1=male  2=female
    'V161342': {
        "template":"I am a XXX.",
        "valmap":{ 1:"man", 2:"woman"},
        "colname": "gender",
        },

#    # income  V161361x
#    '': {
#        "template":"",
#        "valmap":{},
#        },

    # vote_2016 = if_else(V162034 == 2, 0, if_else(as.numeric(V162062x) > 0, as.numeric(V162062x), NA_real_)),
#    '': {
#        "template":"",
#        "valmap":{},
#        },

    # political_interest = if_else(V162256 > 0, V162256, NA_real_),
    'V162256': {
        "template":"I am XXX interested in politics.",
        "valmap":{1:"very", 2:"somewhat", 3:"not very", 4:"not at all"},
        "colname": "political_interest",
        },

    # patriotism = if_else(V162125x > 0, V162125x, NA_real_))
    'V162125x': {
        "template":"It makes me feel XXX to see the American flag.",
        "valmap":{1:"extremely good", 2:"moderately good", 3:"a little good", 4:"neither good nor bad", 5:"a little bad", 6:"moderately bad", 7:"extremely bad"},
        "colname": "patriotism",
        },

    # this is sample address, which may be different than registration address...?
    'V161010d': {
        "template":"I am from XXX.",
        "valmap":fips_state_map,
        "colname": "state",
        },

    }
