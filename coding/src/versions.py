from templatize import Templatizer

prefix = '''Classify the following New York Times article headlines.
Use ONLY the following categories: '''
per_cat_lambda = lambda x: f'"{x}"'
join_cats = ', '
suffix = '\n'
join_inputs = '\n###\n'
input_lambda = lambda x: f'Article Headline: {x}'
# category_lambda = lambda x: f'Category: {x}'
join_input_category = '\nCategory:'

templatizer = Templatizer('nytimes')
output = templatizer.templatize(
    n_exemplars = 4,
    prefix = prefix,
    per_cat_lambda = per_cat_lambda,
    join_cats = join_cats,
    suffix = suffix,
    join_inputs = join_inputs,
    input_lambda = input_lambda,
    join_input_category = join_input_category,
)
breakpoint()
pass