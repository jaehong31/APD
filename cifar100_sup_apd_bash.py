"""
"""

import subprocess
import pdb
import sys
import copy
template = 'python cifar100_apd_run.py \
					--data_type superclass \
					--n_epochs 20\
					--n_classes 5\
					--n_tasks 20\
					--decay_rate 1\
					--mask_hyp 0. \
					--l1_hyp 6e-4 \
					--approx_hyp 100. \
					--clustering_iter 5\
					--k_centroides 2\
					--e_gap_hyp 1e-2 \
					--results_path results/100_superclass'

processes = []
arg_container = []
command = template.format(*arg_container)
process = subprocess.call(command, shell=True)
#process = subprocess.Popen(command, shell=True)
#output = process.wait()
