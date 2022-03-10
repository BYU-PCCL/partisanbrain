# Emotion
All files related to opening the black box.

## Running on SIVRI
1. `ssh remote@sivri`
2. `cd /mnt/pccfs2/backed_up/alexshaw/partisanbrain/partisanbrain/emotions/docker`
3. `git pull`
4. `sh build.sh`
5. `sh run.sh`
6. `pip install --upgrade transformers`
6. `pip install --upgrade torch`
7. `export CUDA_VISIBLE_DEVICES=0,1,2,3`
8. Perform any analysis you need...
9. `exit`
10. Then to move any output files you want locally to your computer using `scp`

## File Descriptions
Below is a description of each file with the latter files mostly for analysis and the earlier files for data collection.

- Anything starting with `nb` is a jupyter notebook for playing around with outputs
- `danger_activations.py` and `sentiment_activations.py` gather and save all of the activations and labels when running sentences through the model
  - SHOULD BE RUN ON GPU
- `neuron_selection.py` is for selecting neurons to manipulate. You also select where to force the values to in this file.
- `probability_analysis.py` is for saving two pieces of data based on manipulated neurons:
  1. Likelihood values of the input sentences
  1. One-token-response distributions (vectors or size `n_vocab`) of the input sentences
  - This output is about 2 GB total for sentiment analysis
  - SHOULD BE RUN ON GPU
- `generate_output.py` is for generating output sentences based on an input prompt and manipulated neurons
  - SHOULD BE RUN ON GPU
- `sentiment_analysis.py` is for analyzing the sentiment of our model generated senteces
  - Designed to work with the output from `generate_output.py`

## Tips for Next Steps
If you want to change the number of neurons or where to push the activations to (e.g. mean vs. extreme) use the `neuron_selection.py` file.

Use `probability_analysis.py` for analytical assessment of neuron manipulation.

Use `generate_output.py` in combination with `sentiment_analysis.py` for qualitative assessment of neuron manipulation.