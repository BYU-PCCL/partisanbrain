import pandas as pd


# Load the imdb experiment data
df = pd.read_pickle("data/imdb/exp_results_gpt3-davinci_23-10-2021.pkl")

# "I think this review was"
# "I think the movie was"

new_token_map = {
    "positive": ["excellent", "awesome", "great",
                 "insightful", "spectacular", "amazingly",
                 "phenomenal", "outstanding", "inspirational",
                 "inspired", "priceless", "terrific",
                 "favorable", "perfect", "happiness",
                 "satisfying", "amazing", "brilliant",
                 "wonderful", "pleasantly", "underrated",
                 "meaningful", "nicely", "enjoyed",
                 "happy", "favourable", "beautiful",
                 "perfectly", "delightful", "entertaining",
                 "marvelous", "perfection", "superb",
                 "pleasant", "fantastic", "flawless",
                 "interesting", "fabulous", "respectable",
                 "remarkable", "masterpiece", "exceptional",
                 "superior"],
    "negative": ["bad", "pathetic", "toxic",
                 "failure", "vile", "disgusting",
                 "worst", "rotten", "shit",
                 "false", "unfavorable", "dislike",
                 "lame", "detrimental", "dreadful",
                 "crappy", "painful", "worse",
                 "dull", "negatively", "nasty",
                 "shitty", "disturbing", "fake",
                 "unpleasant", "bullshit", "hate",
                 "rubbish", "pointless", "disappointing",
                 "meaningless", "weaker", "lacking",
                 "failing", "trash", "incorrect",
                 "incomprehensible", "poor", "terrible",
                 "garbage", "dumb", "junk",
                 "horrible", "worthless", "failed",
                 "crap"]
}

# new_token_map = {
#     "positive": ["excellent", "awesome", "great",
#                  "insightful", "spectacular", "amazingly",
#                  "phenomenal", "outstanding", "inspirational",
#                  "inspired", "priceless", "terrific",
#                  "favorable", "perfect", "happiness",
#                  "satisfying", "amazing", "brilliant",
#                  "wonderful", "pleasantly", "underrated",
#                  "meaningful", "nicely", "enjoyed",
#                  "happy", "favourable", "beautiful",
#                  "perfectly", "delightful", "entertaining",
#                  "marvelous", "perfection", "superb",
#                  "pleasant", "fantastic", "flawless",
#                  "interesting", "fabulous", "respectable",
#                  "remarkable", "masterpiece", "exceptional",
#                  "superior"],
#     "negative": ["bad", "pathetic", "toxic",
#                  "failure", "vile", "disgusting",
#                  "worst", "rotten", "shit",
#                  "false", "unfavorable", "dislike",
#                  "lame", "detrimental", "dreadful",
#                  "crappy", "painful", "worse",
#                  "dull", "negatively", "nasty",
#                  "shitty", "disturbing", "fake",
#                  "unpleasant", "bullshit", "hate",
#                  "rubbish", "pointless", "disappointing",
#                  "meaningless", "weaker", "lacking",
#                  "failing", "trash", "incorrect",
#                  "incomprehensible", "poor", "terrible",
#                  "garbage", "dumb", "junk",
#                  "horrible", "worthless", "failed",
#                  "crap"]
# }

df["token_sets"] = df["token_sets"].apply(lambda x: new_token_map)

df.to_pickle("data/imdb/imdb_post_token_set.pkl")
