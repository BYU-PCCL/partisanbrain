import openai

def run(df, model_size):
    '''
    For each row, run row['prompt'] through gpt-3. Save the response to new column 'response'.
    '''
    responses = []
    for i, row in df.iterrows():
        try:
            prompt = row['prompt']
            response # = openai.Completion.create( # TODO - complete
            responses.append(response)
        else:
            responses.append(None)
    df['response'] = responses
    return df

if __name__ == '__main__':
    # templatize df and run
    pass