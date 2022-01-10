# @vinhowe

from ..survey import Survey, UserInterventionNeededError
from ..constants import DEMOGRAPHIC_COLNAMES

from pathlib import Path
from typing import List, Dict
import pandas as pd
from numpy import nan


MANUAL_FILENAME = "raw.sav"
SURVEY_DATA_PATH = Path("survey_data")
WAVE_67_PATH = SURVEY_DATA_PATH / "pew67" / MANUAL_FILENAME
WAVE_78_PATH = SURVEY_DATA_PATH / "pew78" / MANUAL_FILENAME

DV_NAMES_67 = {
    "ENV2_c_W67": "climate_expanding_coal",
    "ENV2_d_W67": "climate_expanding_wind",
    "ENV2_f_W67": "climate_expanding_solar",
    "CCPOLICY_a_W67": "climate_planting_trees",
    "ENVIR8_e_W67": "climate_is_federal_government_effective",
    "EN7_W67": "climate_human_activity_contribution",
    "CLIM9_W67": "climate_affecting_local_community",
    "RQ1_F1A_W67": "physical_health_medical_scientists",
    "PQ1_F2A_W67": "physical_health_medical_doctors",
    "CLIN_TRIAL1_W67": "physical_health_clinical_trials",
}
DV_NAMES_78 = {
    "ECON1_W78": "economics_conditions_in_country",
    "ECON1B_W78": "economics_conditions_in_country_in_year",
    "SATIS_W78": "politics_satisfied",
    "VTADMIN_POST_US_W78": "politics_election_run_well",
    "ELECTNTFOL_W78": "politics_closely_followed_election_results",
    "COVID_2ASSISTLD_W78": "politics_assistance_package_necessary",
    "POL12_W78": "politics_party_relations_outlook",
    "COVID_OPENMORE_W78": "politics_local_covid_restrictions",
    "DIVISIONSCONC_W78": "politics_party_divisions_concern",
    "VOTELIST_US_W78": "politics_more_voting_better",
}
SINGLE_VALUE_DEMOGRAPHIC_COLNAMES = {
    "F_AGECAT": "age",
    "F_CREGION": "region",
    "F_MARITAL": "marital_status",
}
# Note that the order of the tuples in the data will reflect the ordering here
MULTI_VALUE_DEMOGRAPHIC_COLNAMES = {
    "gender": ["F_SEX", "F_GENDER"],
    "party": ["F_PARTY_FINAL", "F_PARTYLN_FINAL", "F_PARTYSUM_FINAL", "F_PARTYSUMIDEO"],
    "ideology": ["F_IDEO", "F_PARTYSUMIDEO"],
    "education": ["F_EDUCCAT", "F_EDUCCAT2"],
    "income": ["F_INCOME", "F_INC_SDT1", "F_INC_TIER2"],
    "religion": ["F_RELIG", "F_RELIGCAT", "F_BORN", "F_ATTEND"],
    "race_ethnicity": ["F_RACECMB", "F_RACETHNMOD", "F_HISP", "F_HISP_ORIGIN"],
}
DV_QUESTIONS = {
    "climate_expanding_coal": "Do you favor or oppose EXPANDING each of the following sources of energy in our country? [COAL]",
    "climate_expanding_wind": "Do you favor or oppose EXPANDING each of the following sources of energy in our country? [WIND]",
    "climate_expanding_solar": "Do you favor or oppose EXPANDING each of the following sources of energy in our country? [SOLAR]",
    "climate_planting_trees": "Do you favor or oppose the following proposals to reduce the effects of global climate change? Planting about a trillion trees around the world to absorb carbon emissions in the atmosphere",
    "climate_is_federal_government_effective": "How much do you think the federal government is doing to reduce the effects of global climage change [TOO MUCH/TOO LITTLE]",
    "climate_human_activity_contribution": "How much do you think human activity, such as the burning of fossil fuels, contributes to global climate change?",
    "climate_affecting_local_community": "How much, if at all, do you think global climate change is currently affecting your local community?",
    "physical_health_medical_scientists": "In general, would you say your view of medical research scientists is...",
    "physical_health_medical_doctors": "In general, would you say your view of medical doctors is...",
    "physical_health_clinical_trials": "Some medical research studies are called clinical trials in which volunteers participate in a study to help test the safety and effectiveness of new treatments, drugs or devices. How important do you think it is to go through the process of conducting clinical trials, even if it will lengthen the time it takes to develop new treatments?",
    "economics_conditions_in_country": "How would you rate economic conditions in this country today?",
    "economics_conditions_in_country_in_year": "A year from now, do you expect that economic conditions in the country as a whole will be... [BETTER/WORSE/SAME]",
    "politics_satisfied": "All in all, are you satisfied or dissatisfied with the way things are going in this country today?",
    "politics_election_run_well": "Do you think the elections this November in the UNITED STATES were run and administered... [LEVEL OF WELLNESS]",
    "politics_closely_followed_election_results": "How closely, if at all, did you follow the results of the presidential election after polls closed on Election Day?",
    "politics_assistance_package_necessary": "As you may know, Congress and President Trump passed a $2 trillion economic assistance package in March in response to the economic impact of the coronavirus outbreak. Do you think another economic assistance package is... [NECESSARY/NOT NECESSARY]",
    "politics_party_relations_outlook": "Do you think relations between Republicans and Democrats in Washington will get better in the coming year, get worse, or stay about the same as they are now?",
    "politics_local_covid_restrictions": "Thinking about restrictions on public activity because of the coronavirus outbreak IN YOUR AREA, do you think there should be... [MORE/LESS/SAME RESTRICTIONS]",
    "politics_party_divisions_concern": "How concerned, if at all, are you about divisions between Republicans and Democrats?",
    "politics_more_voting_better": "Which statement comes closer to your views, even if neither is exactly right? [WHETHER USA WOULD BE BETTER OFF IF MORE PEOPLE VOTED]",
}


class PewSurvey(Survey):
    def __init__(self):
        super().__init__()

    def download_data(self):
        # TODO: We could do a little more work here to show a list of all datasets that still need downloading
        if not WAVE_67_PATH.exists():
            msg = f"You need to login to download Pew American Trends Panel Wave 67 from https://www.pewresearch.org/science/dataset/american-trends-panel-wave-67/ to {WAVE_67_PATH}"
            raise UserInterventionNeededError(msg)
        if not WAVE_78_PATH.exists():
            msg = f"You need to login to download Pew American Trends Panel Wave 78 from https://www.pewresearch.org/politics/dataset/american-trends-panel-wave-67/ to {WAVE_78_PATH}"
            raise UserInterventionNeededError(msg)

        return pd.read_spss(WAVE_67_PATH), pd.read_spss(WAVE_78_PATH)

    @staticmethod
    def _create_row_tuple_builder(cols_in_df: List[str], colnames: List[str]):
        # One time cost per colnames group
        dv_indices = [colnames.index(col) for col in cols_in_df]

        def build_row_tuple(values):
            row_list = [nan] * len(colnames)
            for i, value in zip(dv_indices, values):
                row_list[i] = value
            return tuple(row_list)

        return build_row_tuple

    @classmethod
    def _modify_wave(cls, df: pd.DataFrame, dv_names: Dict[str, str]):
        # Not sure if there's a problem with modifying the base df in place
        mod_df = df.copy()

        # Rename all columns created from one source column
        mod_df.rename(
            columns=SINGLE_VALUE_DEMOGRAPHIC_COLNAMES,
            # Silently fail if key to be renamed is not found in dataset
            errors="ignore",
            inplace=True,
        )
        mod_df.dropna(
            subset=list(SINGLE_VALUE_DEMOGRAPHIC_COLNAMES.values()), inplace=True
        )

        for key, colnames in MULTI_VALUE_DEMOGRAPHIC_COLNAMES.items():
            cols_in_df = list(filter(lambda v: v in mod_df.columns, colnames))
            # This drops only if _all_ of the columns listed are empty
            mod_df.dropna(subset=cols_in_df, how="all", inplace=True)
            # This does a little bit of hacking to make sure that we have null values in the right order
            tuple_builder = cls._create_row_tuple_builder(cols_in_df, colnames)
            mod_df[key] = mod_df[cols_in_df].apply(tuple_builder, axis=1)

        # We don't want to ignore errors here because it could mean that dv_names has incorrect keys
        mod_df.rename(columns=dv_names, errors="raise", inplace=True)

        df_columns = set(df.columns)
        mod_df_columns = set(mod_df.columns)
        # Just drop all of the columns that were in the original dataframe.
        # We want to exclude renamed columns. Which are columns that are in df but not mod_df.
        # That difference is represented by df.columns - mod_df.columns. This is hacky but I'm fine with it.
        mod_df.drop(
            columns=list(df_columns - (df_columns - mod_df_columns)), inplace=True
        )

        return mod_df

    def modify_data(self, dfs: List[pd.DataFrame]):
        # Note that allowing dfs to be multiple dataframes here relies on
        # the superclass being implemented in such a way that the value returned
        # from download_data is supplied unchanged to modify_data
        df67, df78 = dfs
        df67_processed = self._modify_wave(df67, DV_NAMES_67)
        df78_processed = self._modify_wave(df78, DV_NAMES_78)
        return pd.concat([df67_processed, df78_processed])

    def get_dv_questions(self):
        return DV_QUESTIONS


if __name__ == "__main__":
    # Make sure this runs without errors after pulling the most
    # recent code from GitHub. See surveys/example.py for more
    # information on making your subclass.
    s = PewSurvey()
