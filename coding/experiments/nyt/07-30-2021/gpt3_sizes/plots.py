import sys
sys.path.append('src')
from analysis import ExperimentResults
import os

experiment_directory = 'experiments/nyt/07-30-2021/gpt3_sizes/'
er = ExperimentResults(experiment_directory, ends_with='.pickle', normalize_marginal=True)
# er = ExperimentResults(experiment_directory, ends_with='.pickle', normalize_marginal=False)
# check if plots is subdirectory of experiment directory
plot_dir = os.path.join(experiment_directory, 'plots/')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# plot 'version' as x_variable
er.plot(x_variable='model', save_path=plot_dir)