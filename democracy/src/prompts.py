from abc import ABC, abstractmethod
from collections import defaultdict

import openai
import os
import pandas as pd
import tqdm

# Authenticate with openai
openai.api_key = os.getenv("GPT_3_API_KEY")


QUOTES = [
    ("Pope Francis said, \"We cannot tolerate or "
     "turn a blind eye to racism and exclusion in "
     "any form. At the same time, we have to recognize "
     "that violence is self-destructive and self-defeating. "
     "Nothing is gained by violence and so much is lost. "
     "Let us pray for reconciliation and peace.\""),
    ("Gandhi said, \"Power is of two kinds. One is obtained "
     "by the fear of punishment and the other by acts of love. "
     "Power based on love is a thousand times more effective and "
     "permanent then the one derived from fear of punishment.\""),
    ("Gandhi said, \"True love is boundless like the ocean "
     "and rising and swelling within one spreads itself out "
     "and crossing all boundaries and frontiers envelops "
     "the whole world.\""),
    ("The Bible says, \"Ye have heard that it hath been said, "
     "Thou shalt love thy neighbour, and hate thine enemy. But "
     "I say unto you, Love your enemies, bless them that curse you, "
     "do good to them that hate you, and pray for them which "
     "despitefully use you, and persecute you; That ye may be "
     "the children of your Father which is in heaven: for he "
     "maketh his sun to rise on the evil and on the good, "
     "and sendeth rain on the just and on the unjust.\""),
    ("The Bible says, \"There is no fear in love; but perfect love "
     "casteth out fear: because fear hath torment. He that feareth "
     "is not made perfect in love. We love him [God], because "
     "he first loved us.\""),
    ("C.S. Lewis said, \"It is a serious thing to live in a "
     "society of possible gods and goddesses, to remember that "
     "the dullest and most uninteresting person you talk to may "
     "one day be a creature which, if you saw it now, you would "
     "be strongly tempted to worship, or else a horror and a "
     "corruption such as you now meet, if at all, only in a "
     "nightmare. ... It is in the light of these overwhelming "
     "possibilities, it is with the awe and circumspection proper "
     "to them, that we should conduct all our dealings with one "
     "another, all friendships, all loves, all play, all politics. "
     "There are no ordinary people. You have never "
     "talked to a mere mortal.\""),
    ("The Book of Mormon says, \"And now, my sons, I speak unto "
     "you these things for your profit and learning; for there is "
     "a God, and he hath created all things, both the heavens and "
     "the earth, and all things that in them are, both things to "
     "act and things to be acted upon. ... Wherefore, the Lord God "
     "gave unto man that he should act for himself. Wherefore, man "
     "could not act for himself save it should be that he was enticed "
     "by the one or the other. Wherefore, men are free according to "
     "the flesh; and all things are given them which are expedient "
     "unto man. And they are free to choose liberty and eternal life, "
     "through the great Mediator of all men, or to choose captivity "
     "and death, according to the captivity and power of the devil; "
     "for he seeketh that all men might be miserable "
     "like unto himself.\""),
    ("The Book of Mormon says, \"And now, my sons, remember, "
     "remember that it is upon the rock of our Redeemer, who is "
     "Christ, the Son of God, that ye must build your foundation; "
     "that when the devil shall send forth his mighty winds, yea, "
     "his shafts in the whirlwind, yea, when all his hail and his "
     "mighty storm shall beat upon you, it shall have no power over "
     "you to drag you down to the gulf of misery and endless wo, "
     "because of the rock upon which ye are built, which is a sure "
     "foundation, a foundation whereon if men build they cannot fall.\""),
    ("The Bible says, \"For God so loved the world, that he "
     "gave his only begotten Son, that whosoever believeth "
     "in him should not perish, but have everlasting life.\""),
    ("The Bible says, \"A new commandment I give unto you, "
     "That ye love one another; as I have loved you, that "
     "ye also love one another. By this shall all men "
     "know that ye are my disciples, if ye have love one to another.\""),
    ("The Koran says, \"O mankind! Reverence your Guardian-Lord, "
     "who created you from a single soul, created, of like nature, "
     "the mate, and from them twain scattered (like seeds) "
     "countless men and women;- reverence Allah, through whom "
     "ye demand your mutual (rights), and (reverence) the wombs "
     "(That bore you): for Allah ever watches over you.\""),
    ("The Koran says, \"He created you (all) from a single "
     "person: then created, of like nature, his mate; and "
     "he sent down for you eight head of cattle in pairs: "
     "He makes you, in the wombs of your mothers, in stages, "
     "one after another, in three veils of darkness. such is Allah, "
     "your Lord and Cherisher: to Him belongs (all) dominion. "
     "There is no god but He: then how are ye turned away "
     "(from your true Centre)?\""),
    ("Pope Francis said \"Human dignity is the same for all "
     "human beings: when I trample on the dignity of another, "
     "I am trampling on my own.\""),
    ("Anton Chekhov said, \"Everything on earth is beautiful, "
     "everything -- except what we ourselves think and do when "
     "we forget the higher purposes of life and our "
     "own human dignity.\""),
    ("Helen Keller said, \"If we make up our minds that "
     "this is a drab and purposeless universe, it will be "
     "that, and nothing else. On the other hand, if we believe "
     "that the earth is ours, and that the sun and moon hang in "
     "the sky for our delight, there will be joy upon the hills and "
     "gladness in the fields because the Artist in our souls "
     "glorifies creation. Surely, it gives dignity to life to "
     "believe that we are born into this world for noble ends, "
     "and that we have a higher destiny than can be "
     "accomplished within the narrow limits of this physical life.\""),
    ("In January 2021, many Republicans and then-U.S. President "
     "Donald Trump supporters rioted and stormed the US Capitol. "
     "GOP House Minority Leader Kevin McCarthy said, \"We can "
     "disagree but we should not take it to this level. This is "
     "unacceptable.... You do not do what is happening right now. "
     "People are being hurt. This is unacceptable.\""),
    ("In January 2021, many Republicans and then-U.S. President "
     "Donald Trump supporters rioted and stormed the US Capitol. "
     "GOP House Minority Whip Steve Scalise said, \"United States "
     "Capitol Police saved my life, attacks on law enforcement "
     "officers trying to do their jobs are never acceptable. "
     "Period. We can passionately protest without being violent.\""),
    ("In January 2021, many Republicans and then-U.S. President "
     "Donald Trump supporters rioted and stormed the US Capitol. "
     "Senator Ted Cruz, a Texas Republican, said, \"Violence is "
     "always unacceptable. Even when passions run high. Anyone "
     "engaged in violence—especially against law enforcement—should be "
     "fully prosecuted,\" Cruz wrote on Twitter. \"God bless the "
     "Capitol Police and the honorable men & women of law "
     "enforcement who show great courage keeping all of us safe.\""),
    ("In January 2021, many Republicans and then-U.S. President "
     "Donald Trump supporters rioted and stormed the US Capitol. "
     "President Trump said, \"I want to be very clear: I "
     "unequivocally condemn the violence that we saw last week. "
     "Violence and vandalism have absolutely no place in our "
     "country and no place in our movement. Making America "
     "Great Again has always been about defending the rule of law.\""),
    ("In January 2021, many Republicans and then-U.S. President "
     "Donald Trump supporters rioted and stormed the US Capitol. \"I "
     "think it's dangerous,\" Cheney said. \"I think that we "
     "have to recognize how quickly things can unravel. We "
     "have to recognize what it means for the nation to have a "
     "former president who has not conceded and who continues to "
     "suggest that our electoral system cannot function, "
     "cannot do the will of the people.\""),
    ("Following the murder of George Floyd from police brutality, "
     "many Democrats participated in the Black Lives Matter movement "
     "and protested police brutality violently. Former Vice President "
     "Joe Biden said, \"Protesting such brutality is right and "
     "necessary. It's an utterly American response. But burning "
     "down communities and needless destruction is not. Violence "
     "that endangers lives is not. Violence that guts and "
     "shutters businesses that serve the community is not.\""),
    ("In 2017 and in response to right-wing comments, an Antifa "
     "group in Berkeley, California protested violently and caused "
     "destruction around the community. House Minority Leader Nancy "
     "Pelosi said, \"The violent actions of people calling themselves "
     "antifa in Berkeley this weekend deserve unequivocal "
     "condemnation, and the perpetrators should be "
     "arrested and prosecuted.\""),
    ("In June 2020, Representative James Clyburn of South Carolina, "
     "the House majority whip, said violence related to the Black "
     "Lives Matter movement was hijacking the social justice "
     "movement. \"Peaceful protest is our game. Violence is their "
     "game. Purposeful protest is our game. This looting and "
     "rioting, that's their game. We cannot allow ourselves "
     "to play their game,\" he said."),
    ("In 2020, the popular Black Lives Matter social movement "
     "burst into political violence on several occasions. Former "
     "U.S. President Barack Obama said, \"Let's not excuse "
     "violence, or rationalize it, or participate in it. If "
     "we want our criminal justice system, and American "
     "society at large, to operate on a higher ethical "
     "code, then we have to model that code ourselves.\""),
    ("In August 2020, Black Lives Matter protests became "
     "destructive in Denver, CO. Colorado's Democratic "
     "Governor Jared Polis criticized Denver protesters "
     "who started fires, destroyed property and injured a "
     "police officer, tweeting they had committed \"acts "
     "of criminal terrorism.\" He said, \"An attack against "
     "any of our lives and property is an attack against "
     "all of our lives and property.\""),
    ("In 2020, the popular Black Lives Matter social "
     "movement burst into political violence on several "
     "occasions. Senator Kamala Harris said, \"We must "
     "always defend peaceful protest and peaceful protesters. "
     "We should not confuse them with those looting and "
     "committing acts of violence, including the shooter "
     "who was arrested for murder. Make no mistake, we will "
     "not let these vigilantes and extremists derail "
     "the path to justice.\""),
    ("In 2020, the popular Black Lives Matter social "
     "movement burst into political violence on several "
     "occasions. Former U.S. President Joe Biden said, \"The "
     "deadly violence we saw overnight in Portland is "
     "unacceptable...as a country we must condemn the "
     "incitement of hate and resentment that led to this "
     "deadly clash. It is not a peaceful protest when you "
     "go out spoiling for a fight.\""),
    ("In January 2021, many Republicans and then-U.S. "
     "President Donald Trump supporters rioted and stormed "
     "the US Capitol. Former U.S. President George W. Bush "
     "said, \"I am appalled by the reckless behavior of "
     "some political leaders since the election and by the "
     "lack of respect shown today for our institutions, our "
     "traditions, and our law enforcement. The violent assault "
     "on the Capitol – and disruption of a "
     "Constitutionally-mandated meeting of Congress – was "
     "undertaken by people whose passions have been inflamed "
     "by falsehoods and false hopes. Insurrection could do "
     "grave damage to our Nation and reputation. In the United "
     "States of America, it is the fundamental responsibility "
     "of every patriotic citizen to support the rule of law. "
     "To those who are disappointed in the results of the "
     "election: Our country is more important than the "
     "politics of the moment. Let the officials elected "
     "by the people fulfill their duties and represent "
     "our voices in peace and safety.\""),
    ("In January 2021, many Republicans and then-U.S. "
     "President Donald Trump supporters rioted and stormed "
     "the US Capitol. Senator Josh Hawley said, \"I hope "
     "that this body will not miss the opportunity to take "
     "affirmative action to address the concerns of so "
     "many millions of Americans to say to millions of "
     "Americans tonight that violence is never warranted, "
     "that violence will not be tolerated, those that "
     "engage in it will be prosecuted but that this body "
     "will act to address the concerns of all "
     "Americans across the country.\""),
    ("In January 2021, many Republicans and then-U.S. "
     "President Donald Trump supporters rioted and stormed "
     "the US Capitol. Utah Governor Spencer Cox "
     "said, \"I'm deeply troubled at the chaos, the "
     "devastation, the cowardly acts of violence that "
     "we are seeing in our nation's capital on this day. "
     "As patriots, … as Americans, as people who care deeply "
     "about each other, and care deeply about this great "
     "nation, I urge you to stand up and speak out against "
     "the violence, against the terrorists, against the "
     "evil we have seen in our nations capitol "
     "today. ... We are better than this in America. We "
     "have been an example to the rest of the world when "
     "it comes to our elections when it comes to the way that "
     "we protest when it comes to the freedoms and the "
     "freedoms and liberties that we so enjoy.\""),
    ("In 2020, the popular Black Lives Matter social "
     "movement burst into political violence on several "
     "occasions. Former U.S. President Barack Obama "
     "said, \"The small minority of folks who've "
     "resorted to violence in various forms, whether out "
     "of genuine anger or mere opportunism, are putting "
     "innocent people at risk, compounding the destruction "
     "of neighborhoods that are often already short on "
     "services and investment and detracting from the "
     "larger cause. I saw an elderly black woman "
     "being interviewed today in tears because the "
     "only grocery store in her neighborhood had been "
     "trashed. If history is any guide, that store may "
     "take years to come back. So let's not excuse "
     "violence, or rationalize it, or participate in it. "
     "If we want our criminal justice system, and American "
     "society at large, to operate on a higher "
     "ethical code, then we have to model that "
     "code ourselves.\"")
]


FILLERS = {
    "[P]": {"republican": "Republican",
            "democrat": "Democratic"},
    "[Ps]": {"republican": "Republicans",
             "democrat": "Democrats"},
    "[Pp]": {"republican": "Republican",
             "democrat": "Democrat"},
    "[Po]": {"republican": "Democratic",
             "democrat": "Republican"},
    "[Pso]": {"republican": "Democrats",
              "democrat": "Republicans"},
    "[Ppo]": {"republican": "Democrat",
              "democrat": "Republican"}
}


class Question:

    def __init__(self, text_with_filler, possible_answers):
        self._text_with_filler = text_with_filler
        self._possible_answers = possible_answers

    def get_text(self, party):
        assert party in ["republican", "democrat"]

        text = self._text_with_filler

        for filler in FILLERS:
            text = text.replace(filler, FILLERS[filler][party])

        return text

    def get_possible_answers(self):
        return self._possible_answers


QUESTIONS = [
    Question(("If asked whether [Ps] should \"reduce\" "
              "or \"maintain\" the number of polling "
              "stations in areas that support [Pso], "
              "I would say that [Ps] should \""),
             ["reduce", "maintain"]),
    Question(("If asked whether [P] governors should "
              "\"ignore\" or \"heed\" unfavorable "
              "court rulings by [Po]-appointed judges, "
              "I would say that [P] governors should \""),
             ["ignore", "heed"]),
    Question(("If asked whether [P] governors should "
              "\"prosecute\" or \"tolerate\" journalists who "
              "accuse [P] politicians of misconduct without "
              "revealing sources, I would say that [P] "
              "governors should \""),
             ["prosecute", "tolerate"]),
    Question(("If asked whether [Ps] should \"accept\" "
              "or \"deny\" the results of elections if "
              "they lose, I would say that [Ps] "
              "should \""),
             ["deny", "accept"]),
    Question(("If asked whether it is \"always\" or "
              "\"never\" OK for a [Pp] to send "
              "threatening and intimidating messages "
              "to [Po] party leaders, I would answer by "
              "saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "OK for an ordinary [Pp] in the public to harass "
              "an ordinary [Ppo] on the Internet, in a way that "
              "makes the target feel frightened, I would "
              "answer by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "justified for [Ps] to use violence in advancing "
              "their political goals these days, I would answer "
              "by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether it is \"always\" or \"never\" "
              "justified for [Ps] to use violence if the [Po] "
              "party wins more races in the next election, "
              "I would answer by saying that it is \""),
             ["always", "never"]),
    Question(("If asked whether I feel \"warm\" or \"cold\" "
              "towards [Pso], I would say that I feel \""),
             ["warm", "cold"]),
    Question(("If I were given a dollar and then asked "
              "whether I would share \"all of it\" or "
              "\"none of it\" with a [Ppo], I would "
              "answer that I would share \""),
             ["none of it", "all of it"])
]


class PromptProcessor:

    def __init__(self, engine):
        possible_engines = ["davinci", "curie", "babbage", "ada"]
        assert engine in possible_engines, f"{engine} is not a valid engine"
        self._engine = engine

    def process_prompt(self, prompt, max_tokens=1):
        try:
            return openai.Completion.create(engine=self._engine,
                                            prompt=prompt,
                                            max_tokens=max_tokens,
                                            logprobs=100)
        # TODO: Catch more specific exception here
        except Exception as e:
            print(f"Exception in method process_prompt: {e}")
            return None


class Prompt(ABC):

    def __init__(self, respondent_data, party, question_idx):
        assert party in ["republican", "democrat"]
        self._row = respondent_data
        self._question_idx = question_idx
        self._party = party

    def _get_backstory(self):

        backstory = []

        # Party
        party_dict = {
            "Democrat": "Democrat",
            "Republican": "Republican"
        }
        backstory.append(f"I am a {party_dict[self._row.party]}.")

        # Gender
        gender_dict = {
            "male": "male",
            "female": "female"
        }
        backstory.append(f"I am {gender_dict[self._row.gender]}.")

        # Race
        race_dict = {
            "White": "White",
            "Black": "Black",
            "Hispanic": "Hispanic",
            ("Asian/Native Hawaiian/"
             "Pacific Islander"): "Asian American/Pacific Islander",
            "Native American/Alaskan Native": "Native American"
        }
        backstory.append(f"I am {race_dict[self._row.race]}.")

        # Age
        backstory.append(f"I am {self._row.age} years old.")

        # Education
        educ_dict = {
            "no high school degree": "I never graduated from high school.",
            "high school graduate": "I graduated from high school.",
            "some college": "I completed some college.",
            "bachelor's degree": "I earned a bachelor's degree.",
            "graduate degree": "I earned a graduate degree."
        }
        backstory.append(educ_dict[self._row.education])

        return " ".join(backstory)

    @abstractmethod
    def _get_treatment(self):
        pass

    def get_question(self):
        return QUESTIONS[self._question_idx]

    def get_prompt(self):
        prompt = self._get_backstory() + "\n\n"
        if self._get_treatment() is not None:
            prompt += self._get_treatment() + "\n\n"
        prompt += self.get_question().get_text(self._party)
        return prompt


class ContextPrompt(Prompt):

    def __init__(self, respondent_data, party, question_idx, context):
        super().__init__(respondent_data, party, question_idx)
        self._context = context

    def _get_treatment(self):
        return self._context


def run_experiment(republican_csv_fname,
                   democrat_csv_fname,
                   engine,
                   context_fn,
                   treatment,
                   dry_run=False):

    party_fnames = {"republican": republican_csv_fname,
                    "democrat": democrat_csv_fname}

    for party_name, fname in party_fnames.items():

        # Load data
        df = pd.read_csv(fname)

        # Set up processing
        proc = PromptProcessor(engine=engine)

        # Loop through
        results = defaultdict(list)
        for _, row in tqdm.tqdm(df.iterrows(),
                                total=df.shape[0]):
            for dv_idx in tqdm.tqdm(range(10)):

                # Get context here
                contexts = context_fn()

                for context in contexts:

                    # Process prompt made from the context
                    prompt = ContextPrompt(row,
                                           party_name,
                                           dv_idx,
                                           context=context)
                    if not dry_run:
                        api_resp = proc.process_prompt(prompt.get_prompt())
                        question = prompt.get_question()
                        possible_answers = question.get_possible_answers()
                        results["prompt"].append(prompt.get_prompt())
                        results["possible_answers"].append(possible_answers)
                        results["api_resp"].append(api_resp)
                        results["gender"].append(row.gender)
                        results["race"].append(row.race)
                        results["age"].append(row.age)
                        results["education"].append(row.education)
                        results["treatment"].append(treatment)
                        results["dv_idx"].append(dv_idx)
                        results["party"].append(party_name)
                    else:
                        print(prompt.get_prompt())
                        print("=" * 50)

            if not dry_run:
                out_fname = f"{treatment}_{party_name}.csv"
                pd.DataFrame.from_dict(results).to_csv(out_fname,
                                                       index=False)


def run_passive_experiment(republican_csv_fname,
                           democrat_csv_fname,
                           engine,
                           treatment="passive",
                           dry_run=False):
    run_experiment(republican_csv_fname,
                   democrat_csv_fname,
                   engine,
                   treatment=treatment,
                   context_fn=lambda: [None],
                   dry_run=dry_run)


def get_gpt_reflection(prompt, processor):
    resp_obj = processor.process_prompt(prompt, max_tokens=200)
    resp_str = resp_obj.choices[0].text
    resp_str = resp_str.replace("  ", " ")
    resp_str = resp_str.split("\n")[0]
    # resp_str = resp_str.split(".")[:-1]
    return resp_str


def run_kalmoe_experiment(republican_csv_fname,
                          democrat_csv_fname,
                          engine,
                          treatment="kalmoe",
                          dry_run=False):
    run_experiment(republican_csv_fname,
                   democrat_csv_fname,
                   engine,
                   treatment=treatment,
                   context_fn=lambda: QUOTES,
                   dry_run=dry_run)


def run_mixed_affect_experiment(republican_csv_fname,
                                democrat_csv_fname,
                                engine,
                                treatment="mixed_affect",
                                dry_run=False):

    pass


if __name__ == "__main__":
    # run_kalmoe_experiment("best_republican_sample_sml.csv",
    #                       "best_democrat_sample_sml.csv",
    #                       "ada",
    #                       dry_run=False)
    run_passive_experiment("best_republican_sample.csv",
                           "best_democrat_sample.csv",
                           "ada",
                           dry_run=True)
    # run_mixed_affect_experiment("best_republican_sample.csv",
    #                             "best_democrat_sample.csv",
    #                             "ada",
    #                             dry_run=True)

#     prompt = '''The other day, I read that "When reading messages about political opponents, many people feel mixed emotions because the messages often challenge some beliefs while affirming others. Important research from psychology shows that when people feel such mixed emotions, their unconscious reaction is often to resolve their discomfort by doubling down on their initial beliefs even more and becoming less open to opposing perspectives. But research also shows that simply being mindful of these mixed reactions can change us, allowing us to focus on what we have in common and find ways to work together, even across lines of difference.
# In the box below, please briefly describe a time when you felt mixed emotion in response to a message about political opponents, but consciously resolved those feelings by focusing on what you have in common with those people:"
# In my response, I wrote that these thoughts'''

#     print(get_gpt_reflection(prompt, PromptProcessor(engine="davinci")))
