# A short analysis on US Census Data

- Notebooks are found in jnb folder
- Code is found in src folder
- Documentation and slides in docs folder

In order to run the code it is reccommended to install the requirements in requirements.txt file.

EDA and experiments are found in jupyter notebooks files. These levarage code from the scripts in src folder.

When running the code with the current config (can be change in config.py) it is necessary to run the code from the root folder of the project.
In order to run the jupyter notebooks it might be necessary to change the PATH variables in config file

A new model can be trained on the provided train dataset by executing src/main.py.
Doing so will also test the obtained model against the test set.
The obtained model will be automatically saved in ./models folder
