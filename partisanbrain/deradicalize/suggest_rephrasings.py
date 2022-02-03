# Import numpy, pandas, and the rest of the modules
import numpy as np
import pandas as pd


# First import the data

path = "data/guncontrol_convos.xlsx"
df = pd.read_excel(path)
parties = []
messages = []
running_convo = ""
instructions = ("The following is an exchange between a Republican and a "
                "Democrat. They are talking about gun control.\n\n")
running_convo += instructions

# Different warnings for different DVs.

# Conversation again
# warning_string = ("This might be a counterproductive thing to say, if these "
# "two ever hope to talk again. ")

# More sympathetic to each other's point of view
warning_string = ("This might be a counterproductive thing to say, if these "
                  "two are to ever see each others' points of view. ")

# Non-violence
# warning_string = ("This might be a counterproductive thing to say, if we want "
#                   "these two to not support political violence or a civil war in the next decade. ")

rephrasing_specs = {
    "paraphrase_and_ask":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              f"paraphrases what the {opp_party} said last, and then asks a clarifying "
                              f"question or question of understanding? (NOT a 'trap' question or a "
                              f"question to make a point)"),

    "rephrase_to_i":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              "makes a statement beginning with 'I feel that...' or 'It seems to me that' "
                              "instead of making braod, generalizing claims?"),

    "restate_before_disagree":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              f"restates, in their own words, the {opp_party}'s last message before "
                              "introducing disagreement?"),

    "empathy":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              f"emphasizes shared beliefs and values with the {opp_party}, "
                              "essentially indicating that we all care about the same big-picture things "
                              "rather than emphasizing the disagreement in the details?"),

    "vulnerability":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              "emphasizes self-disclosure, ",
                              "personal stories, humor, and social trust as a way to improve their conversation "
                              "with the {opp_party} and decision-making? The idea is that vulnerability increases "
                              "trust and understanding among others, which seems crucial when having "
                              "disagreements and discussions."),

    "translate_values":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} "
                              "still communicates their position, but motivates it with different "
                              "underlying concerns? It also doesn't necessarily have to emphasize that "
                              "we all believe the same thing."),

    "perspective_taking":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              f"expresses what they think it would be like to be in {opp_party}'s "
                              "position, and then reaffirm the original message?"),


    "perspective_getting":
    lambda party, opp_party: (warning_string +
                              f"Can you suggest a rephrasing of that message, where the {party} instead "
                              f"actively requests the view of {opp_party} in a non-judgmental and "
                              "listening way?"),
}
for theory, spec in rephrasing_specs.items():
    for ix, row in df.iterrows():
        party = {"D": "Democrat", "R": "Republican"}[row.Party]
        opp_party = {"D": "Republican", "R": "Democrat"}[row.Party]
        message = row.Message
        parties.append(party)

        messages.append(message)
        prompt = (
            running_convo
            + f"\nNow the {party} wants to say '{message}'.\n\n{spec(party, opp_party)}\n\nHere is the rephrasing:"
        )
        running_convo += f"{party}: {message}\n"
    print(prompt)
    breakpoint()


breakpoint()
breakpoint()
