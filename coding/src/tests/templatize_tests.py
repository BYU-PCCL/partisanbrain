from templatize import Templatizer

def test_ns_exemplars():
    """Tests whether each n_exemplar in n_exemplars shows up 28 times, and that there are exactly 2"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_exemplars = [1, 3]
    output = templatizer.templatize_many(
        ns_exemplars=ns_exemplars,
        )
    counts = output.n_exemplars.value_counts()
    assert len(counts) == 2
    assert output.n_exemplars.value_counts().loc[3.0] == 28
    assert output.n_exemplars.value_counts().loc[1.0] == 28

                
def test_ns_per_category():
    """Tests whether each code appears np.sum(ns_per_category) times"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_per_category = [1, 2]
    output = templatizer.templatize_many(
        ns_per_category=ns_per_category,
        )
    counts = output.topic_2digit.value_counts().unique()
    assert len(counts) == 1
    assert counts[0] == np.sum(ns_per_category)


    ns_per_category = [1, 2, 4]
    output = templatizer.templatize_many(
        ns_per_category=ns_per_category,
        )
    counts = output.topic_2digit.value_counts().unique()
    assert len(counts) == 1
    assert counts[0] == np.sum(ns_per_category)

def test_exemplar_constancy():
    """Test whether marginal exemplar was the only one that changed"""
    templatizer = Templatizer(dataset_name='nytimes')
    ns_exemplars = [1, 2, 3]
    output = templatizer.templatize_many(
        ns_exemplars=ns_exemplars,
        )
    articledf = output[output.article_id==4262] 
    article1shotexemplars = articledf.iloc[0].exemplars.split('|||')
    article2shotexemplars  = articledf.iloc[1].exemplars.split('|||')
    article3shotexemplars  = articledf.iloc[2].exemplars.split('|||')
    assert article3shotexemplars[0] == article2shotexemplars[0] == article1shotexemplars[0]
    assert article3shotexemplars[1] == article2shotexemplars[1]

def test_instance_seed():
    """Test whether instance set was the same for all instances, and different for a different seed"""

    templatizer = Templatizer(dataset_name='nytimes')
    n_per_category = 1
    seed_instances = 42

    output1 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_instances=seed_instances,
        )
    output2 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_instances=seed_instances,
        )

    output3 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_instances=seed_instances+1,
    )

    assert output1.iloc[0][templatizer.args['input_column']] == output2.iloc[0][templatizer.args['input_column']]
    assert output1.iloc[1][templatizer.args['input_column']] == output2.iloc[1][templatizer.args['input_column']]
    assert output1.iloc[2][templatizer.args['input_column']] == output2.iloc[2][templatizer.args['input_column']]
    assert output1.iloc[0][templatizer.args['input_column']] != output3.iloc[0][templatizer.args['input_column']]
    assert output2.iloc[0][templatizer.args['input_column']] != output3.iloc[0][templatizer.args['input_column']]

def test_exemplar_seed():
    """Test whether exemplar set was the same for all instances, and different for a different seed"""

    templatizer = Templatizer(dataset_name='nytimes')
    n_per_category = 1
    seed_exemplars = 42

    output1 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_exemplars=seed_exemplars,
    )
    output2 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_exemplars=seed_exemplars,
    )
    output3 = templatizer.templatize(
        n_per_category=n_per_category,
        seed_exemplars=seed_exemplars+1,
    )

    assert output1.iloc[0]['exemplars'] == output2.iloc[0]['exemplars']
    assert output1.iloc[0]['exemplars'] != output3.iloc[0]['exemplars']
    assert output2.iloc[0]['exemplars'] != output3.iloc[0]['exemplars']

def test_kwargs():
    """Test whether kwargs are passed through"""
    prefix = 'uSiNg OnLy ThE fOlLoWiNg CaTeGoRiEs'
    per_cat_lamda = 'per cat lambda'
    join_cats = 'join cats'
    suffix = 'suffix'
    input_lambda = 'input lambda'
    category_lambda = 'category lambda'
    join_input_category = 'join input category'
    join_inputs = 'join inputs'

    templatizer = Templatizer(dataset_name='nytimes')
    output = templatizer.templatize_many(
        ns_exemplars=[1, 2],
        prefix=prefix,
        per_cat_lambda = lambda x: per_cat_lamda,
        join_cats = join_cats,
        suffix = suffix,
        input_lambda = lambda x: input_lambda,
        category_lambda = lambda x: category_lambda,
        join_input_category = join_input_category,
        join_inputs = join_inputs,
    )

    assert prefix in output.iloc[0].prompt
    assert per_cat_lamda in output.iloc[0].prompt
    assert join_cats in output.iloc[0].prompt
    assert suffix in output.iloc[0].prompt
    assert input_lambda in output.iloc[0].prompt
    assert category_lambda in output.iloc[0].prompt
    assert join_input_category in output.iloc[0].prompt
    assert join_inputs in output.iloc[0].prompt

    output = templatizer.templatize(
        n_exemplars=1,
        prefix=prefix,
        per_cat_lambda = lambda x: per_cat_lamda,
        join_cats = join_cats,
        suffix = suffix,
        input_lambda = lambda x: input_lambda,
        category_lambda = lambda x: category_lambda,
        join_input_category = join_input_category,
        join_inputs = join_inputs,
    )

    assert prefix in output.iloc[0].prompt
    assert per_cat_lamda in output.iloc[0].prompt
    assert join_cats in output.iloc[0].prompt
    assert suffix in output.iloc[0].prompt
    assert input_lambda in output.iloc[0].prompt
    assert category_lambda in output.iloc[0].prompt
    assert join_input_category in output.iloc[0].prompt
    assert join_inputs in output.iloc[0].prompt

def test_templatize_many():
    """Instantiate two Templatizer instances, call templatize_many on both of 
    them, and check whether they generate the same output."""
    templatizer1 = Templatizer(dataset_name='nytimes')
    templatizer2 = Templatizer(dataset_name='nytimes')
    output1 = templatizer1.templatize_many()
    output2 = templatizer2.templatize_many()
    assert output1.equals(output2)

    output1 = templatizer1.templatize_many(
        ns_per_category=[1, 2],
        ns_exemplars=[2, 3],
        n_exemplar_runs=2,
        n_instance_runs=2,
    )
    output2 = templatizer2.templatize_many(
        ns_per_category=[1, 2],
        ns_exemplars=[2, 3],
        n_exemplar_runs=2,
        n_instance_runs=2,
    )
    assert output1.equals(output2)


def test_description():
    """Test whether description is added"""
    # test with descriptions
    templatizer = Templatizer(dataset_name='nytimes')
    output = templatizer.templatize(use_description=True)
    # make sure nyt_descriptions are in the prompt
    for desc in nyt_descriptions.values():
        assert desc in output.iloc[0].prompt
    
    # test without descriptions
    templatizer = Templatizer(dataset_name='nytimes')
    output = templatizer.templatize(use_description=False)
    # make sure nyt_descriptions are not in the prompt
    for desc in nyt_descriptions.values():
        if desc != '':
            assert desc not in output.iloc[0].promp

def tests():

    test_exemplar_constancy()
    test_ns_exemplars()
    test_ns_per_category()
    test_instance_seed()
    test_exemplar_seed()
    test_kwargs()
    test_templatize_many()
    test_description()
    print('Tests all passed!')

if __name__ == '__main__':
    tests()