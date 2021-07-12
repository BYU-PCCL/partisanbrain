# partisanbrain
Repo for everything in the partisan brain project.

## Organization
Each project will have its own folder (persuasion, coding, etc.). All code should be run from the root project folder.

Within each folder, everything should be organized into subdirectories. 
```
partisanbrain
  / persuasion
    / src
    / data
    / experiments
  / coding
    / src
    / data
    / experiments
...
```

#### Subdirectories
- `src`: contains all scripts. Everything here should be backed up to the repo.
- `data`: contains all raw data. Nothing should be backed up here, with the possible exceptions of instructions to download the data.
- `experiments`: contains all experiments and results, including a `run.sh` file that is used to run the experiment. Make sure commit is complete. Should we need to replicate an experiment, we should be able to return to the commit for the given shell file and rerun the experiment to reproduce the results. Only the `run.sh` file should be backed up to the repo.
