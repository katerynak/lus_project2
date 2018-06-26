import subprocess
from os import listdir
from os.path import isfile, join
import itertools
import numpy as np
import os


if __name__== "__main__":

    temp_dir = "./templates/"
    templates = [f for f in listdir(temp_dir) if isfile(join(temp_dir, f))]
    train_files = ["../../data/NLSPARQL.train.pref.pos.data",
                   "../../data/NLSPARQL.train.iob.pref.data",
                   "../../data/NLSPARQL.train.iob.suff.data"]
    test_files = ["../../data/NLSPARQL.test.pref.pos.data",
                  "../../data/NLSPARQL.test.iob.pref.data",
                  "../../data/NLSPARQL.test.iob.suff.data"]
    suffs = ["pref_pos_", "iob_pref", "iob_suff"]
    out_dir = "./pred_data/"
    eval_dir = "./eval_out/"

    freq_cut_off = list(range(3, 6))
    algorithms = ["CRF"]
    cs = list(range(1, 5))

    params = [templates, freq_cut_off, algorithms, cs]
    params = list(itertools.product(*params))
    params = np.array(params)
    iterations = 30
    indices = np.random.randint(0, params.shape[0], iterations)
    params = params[indices]

    for template, freq, alg, c in params:
        for train_file, test_file, suff in zip(train_files, test_files, suffs):
            out_file = out_dir + suff + "crf_learn__a_{}__c_{}__f_{}__m_100_{}".format(alg, c,
                                                                                freq,
                                                                                template,
                                                                                train_file)
            if os.path.isfile(out_file):
                continue

            subprocess.call("crf_learn -a {} -c {} -f {} -p 8 -m 100 {} {} model".format(alg, c,
                                                                                         freq,
                                                                                         temp_dir + template,
                                                                                         train_file), shell=True)

            subprocess.call("crf_test -m model {} >{}".format(test_file, out_file), shell=True)

            eval_file = eval_dir + suff + "crf_learn__a_{}__c_{}__f_{}__m_100_{}".format(alg, c,
                                                                                  freq,
                                                                                  template,
                                                                                  train_file)

            subprocess.call("../conlleval.pl -d '\t' <{} >{}".format(out_file, eval_file), shell=True)
