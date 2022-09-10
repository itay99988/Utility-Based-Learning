# Overview 
”Utility Value Based Learning” is an automata learning algorithm with two different variants.
This work is a Python implementation of the algorithmic approaches described in the paper 
”A Reinforcement-Learning Style Algorithm for Black Box Automata”, written by the following authors: 
Itay Cohen, Roi Fogler, and Prof. Doron Peled (Bar-Ilan University, Israel). 
The paper has been accepted to MEMOCODE 2022 conference. 
We compare our algorithm with an existing automata learning algorithm - RPNI. This repository also contains
a few scripts that can be used to fully reproduce of the experiments conducted in the paper. For more information 
about the algorithm and the experiments, please consult the attached paper.

### Important Files
- utility_based.py - contains both variants of our algorithm: utility based with lookahead, 
and dual clustering.

- RPNI.py and rpni_runner.py - contain our implementation of the RPNI algorithm.

- utility_based_runner.py and rpni_runner.py - experiments reproduction.

# Installation
- Windows users should install graphviz at first:
install graphviz for windows: https://www.graphviz.org/download/
- Install the python packages listed in the requirements.txt file

# Execution Instructions
The following instructions enable a complete reproduction of the experiments.
### Command Line Arguments
The following arguments can be used for both utility_based_runner.py and rpni_runner.py files.

| Argument       | Details                                                                                                                                                                                                                                                                                 |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--mode`       | accepts the value `dual` for the "dual clustering" approach, any other value for the lookahead approach                                                                                                                                                                                 |
| `--repeat`     | indicates the number of repetitions. (accepts any `int` value)                                                                                                                                                                                                                          |
| `--experiment` | possible values:`cross7`, `cross9`, `cross11`, `cross13`, `cross15`, `comb_lock4`, `comb_lock5`, `comb_lock6`, `comb_lock7`, `div4`, `div5`, `div6`, `div7`, `div9`, `parenthesis3`, `parenthesis4`, `parenthesis5`, `parenthesis6`, `comb_div4`, `comb_div6`, `comb_div8`, `comb_div9` |                                                                                            |

### Utility Based Methods Experiments
Run utility_based_runner.py using the above command line arguments to reproduce an experiment of one of our algorithm's variants. 
For example:

`python3 utility_based_runner.py --mode=dual --experiment=comb_div8 --repeat=10`

### RPNI Experiments
Run rpni_runner.py using the above command line arguments (except of 'mode') to reproduce an RPNI experiment.
For example:

`python3 rpni_runner.py --experiment=cross7 --repeat=10`

# Logs
Logs are automatically saved for each utility value based experiment. Inside the "Logs" directory, you will
find a subdirectory name after the experiment. The learned automaton and the total running time are 
documented there.

# Contributors
* [Itay Cohen](https://github.com/itay99988), Bar Ilan University
* Roi Fogler, Bar Ilan University
* [Doron Peled](https://u.cs.biu.ac.il/~doronp/), Bar Ilan University