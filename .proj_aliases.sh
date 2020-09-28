#!/bin/bash

module unload python
module load conda2/4.2.13 cuda/10.2

# Bash aliases used in this project.
alias lpt_srun="srun --pty -p priority --mem 30G -c 3 -t 0-12:00 /bin/bash"
alias lpt_env="conda activate learnpytorch && bash .proj_aliases.sh"
alias lpt_jl="jupyter lab --port=7014 --browser='none'"
alias lpt_sshlab='ssh -N -L 7014:127.0.0.1:7014'
