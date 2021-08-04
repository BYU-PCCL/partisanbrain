import sys
sys.path.append('src')
from analysis import ExperimentResults
import os

experiment_directory = 'experiments/nyt/07-20-2021/EI/'
er = ExperimentResults(experiment_directory, ends_with=['EI.pickle', 'EInex0.pickle'], normalize_marginal=True)
# er = ExperimentResults(experiment_directory, ends_with='.pickle', normalize_marginal=False)
# check if plots is subdirectory of experiment directory
plot_dir = os.path.join(experiment_directory, 'plots/')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

er.plot_category_accuracies(save_path=plot_dir)
er.plot_confusion_matrix(save_path=plot_dir)
er.plot_top_k_accuracies(max_k=28, save_path=plot_dir)
# er.plot(split_by='exemplar_method')

# plot by different colors
er.plot(
    split_by=['exemplar_set_ix', 'instance_set_ix'],
    color_by='instance_set_ix',
    save_path=plot_dir,
)
# plot instances
er.plot(
    split_by = ['instance_set_ix'],
    color_by='instance_set_ix',
    save_path=plot_dir,
)
# plot exemplars
er.plot(
    split_by = ['exemplar_set_ix'],
    color_by='exemplar_set_ix',
    save_path=plot_dir,
)

plot_dir = os.path.join(plot_dir, 'ensemble/')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# with aggregate predictions
df = er.average_predictions()
df_n = er.average_predictions(['n_exemplars'])

er.plot_category_accuracies(df, save_path=plot_dir)
er.plot_confusion_matrix(df, save_path=plot_dir)

# split by instance set and exemplar set
er.plot(df_n, save_path='plots/ensemble/')