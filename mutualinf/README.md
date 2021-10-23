# Bottlenecks before we are to the "only analysis remaining" phase
- [ ] Ensure all subclasses have good token sets with just one word for each ground truth item - ideally sanity check these for each prompt
- [ ] Ensure exactly 20 prompts for each dataset (unless we decide against this)
- [ ] Replace the BoolQ prompts that currently require a violation of that with new prompts and rerun BoolQ
- [ ] Write a script to update the post-experiment pickle files to have correct token sets
- [ ] Make Dataset compatible with special requirements of CommonSenseQA (I'm handling this)
- [ ] Make good templates for CommonSenseQA with associated good token sets
- [ ] Run CommonSenseQA experiment
- [ ] Make a way to manually override the token sets from pickled experiment dataframe

# After all of those bullet points are done
- [ ] Run postprocessing for all datasets (we can use the script for doing it all at once)
