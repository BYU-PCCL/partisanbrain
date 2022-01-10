import pandas as pd
# row: age, gender, race, state

# feel free to do lambda functions (like below) or something else
templatize_info = {
    'anes-voting': {
        # this is just an example, feel free to comment it out. I actually don't know what all the demographics will be yet, Chris will send it over
        'template1': lambda row: f'''I am {row['age']} years old. I am {row['gender']}. I am from {row['state']}. I am {row['race']}. In 2016, I voted for''',
        'template2': lambda row: f'''I am a {row['race']} {row['age']}-year old {row['gender']} from from {row['state']}.  In 2016, I voted for''',
    }
}

def templatize(df, name):
    '''
    Templatize the df containing demographic information for 2016 votes.
    Return a dataframe with all the same columns, plus 'prompt' with the filled out template, and 'template' with the template name.
    Arguments:
        df (pd.DataFrame): dataframe containing demographic information for 2016 votes
        name (str): name of the template to use. Supported: 'anes-voting'
    Returns:
        pd.DataFrame: dataframe with all the same columns, plus 'prompt' with the filled out template, and 'template' with the template name
    '''
    templates = templatize_info[name]
    dfs = []
    for template_name, template_lambda in templates.items():
        df_template = df.copy()
        df_template['prompt'] = df_template.apply(template_lambda, axis=1)
        df_template['template'] = template_name
        dfs.append(df_template)
    # concat
    df_out = pd.concat(dfs)
    return df_out

if __name__ == '__main__':
    example_df = pd.DataFrame({
        'age': [20, 30, 40],
        'gender': ['male', 'female', 'male'],
        'state': ['Utah', 'Texas', 'Vietnam'],
        'race': ['white', 'Black', 'Native American'],
        'ground truth': ['Trump', 'Clinton', 'Trump'],
    })
    # templatize example_df
    example_df = templatize(example_df, 'anes-voting')
    print(example_df)