from time import time
from argparse import ArgumentParser
import importlib
import json
import pickle
import networkx as nx
import itertools
import pdb
import sys
sys.path.insert(0, './')

from gem.utils      import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from gem.evaluation.evaluate_graph_reconstruction import expGR
from gem.evaluation.evaluate_link_prediction import expLP
from gem.evaluation.evaluate_node_classification import expNC
from gem.evaluation.visualize_embedding import expVis
from time import sleep
import matplotlib as mlp
import os

"""The script implements hope_based experiments. It takes input 2 config files, one with name same as dataset, for example if the dataset
named karate, the file is karate.conf and the other params2.conf.
Sample usage
    python gem/experiments/hope_based.py -data sbm -dim 128 -exp gr,lp
If you are running experminets where embedding graphs are plotted on matplotlib, use "export DISPLAY:0" command before running
this python script.
For nodeclassification, node lables are necessary to provide.

""" 
def learn_emb(MethObj, di_graph, params, res_pre, m_summ):
    if params["experiments"] == ["lp"]:
        X = None
    else:
        print('Learning Embedding: %s' % m_summ)
        if not bool(int(params["load_emb"])):
            X, learn_t = MethObj.learn_embedding(graph=di_graph,
                                                 edge_f=None,
                                                 no_python=True)
            print('\tTime to learn embedding: %f sec' % learn_t)
            pickle.dump(X, open('%s_%s.emb' % (res_pre, m_summ), 'wb'))
            pickle.dump(learn_t,
                        open('%s_%s.learnT' % (res_pre, m_summ), 'wb'))
        else:
            X = pickle.load(open('%s_%s.emb' % (res_pre, m_summ),
                                 'rb'))
            try:
                learn_t = pickle.load(open('%s_%s.learnT' % (res_pre, m_summ),
                                           'rb'))
                print('\tTime to learn emb.: %f sec' % learn_t)
            except IOError:
                print('\tTime info not found')
    return X


def run_exps(MethObj, di_graph, data_set, node_labels, params):
    m_summ = MethObj.get_method_summary()
    res_pre = "gem/results/%s" % data_set
    X = learn_emb(MethObj, di_graph, params, res_pre, m_summ)
    if "gr" in params["experiments"]:
        expGR(di_graph, MethObj,
              X, params["n_sample_nodes"],
              params["rounds"], res_pre,
              m_summ, is_undirected=params["is_undirected"])
    if "lp" in params["experiments"]:
        expLP(di_graph, MethObj,
              params["n_sample_nodes"],
              params["rounds"], res_pre,
              m_summ, is_undirected=params["is_undirected"])
    if "nc" in params["experiments"]:
        if "nc_test_ratio_arr" not in params:
            print('NC test ratio not provided')
        else:
            expNC(X, node_labels, params["nc_test_ratio_arr"],
                  params["rounds"], res_pre,
                  m_summ)
    if "viz" in params["experiments"]:
        if MethObj.get_method_name() == 'hope_gsvd':
            d = int(X.shape[1] / 2)
            expVis(X[:, :d], res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)
        else:
            expVis(X, res_pre, m_summ,
                   node_labels=node_labels, di_graph=di_graph)


def call_exps(params, data_set):
    # print('Dataset: %s' % data_set)
    model_hyp = json.load(
        open('gem/experiments/config/%s.conf' % data_set, 'r')
    )
    if params["node_labels"] == True:
        f = open('gem/data/%s/node_labels.pickle' % data_set, 'rb')
        node_labels = pickle.load(f, encoding = 'latin1')
    else:
        node_labels = None
    meth = 'hope'
    di_graph = nx.read_gpickle('gem/data/%s/graph.gpickle' % data_set)
    for d in params["dimensions"]:
        dim = int(d)
        MethClass = getattr(
            importlib.import_module("gem.embedding.%s" % meth),
            'HOPE'
        )
        hyp = {"d": dim, 'data_set': data_set}
        hyp.update(model_hyp[meth])
        MethObj = MethClass(hyp)
        run_exps(MethObj, di_graph, data_set, node_labels, params)


if __name__ == '__main__':

    t1 = time()
    parser = ArgumentParser(description='Graph Embedding Experiments')
    parser.add_argument('-data', '--data_sets',
                        help='dataset names (default: sbm)')
    parser.add_argument('-dim', '--dimensions',
                        help='embedding dimensions list(default: 2^1 to 2^8)')
    parser.add_argument('-meth', '--methods',
                        help='method list (default: all methods)')
    parser.add_argument('-exp', '--experiments',
                        help='exp list (default: gr,lp,viz,nc)')
    parser.add_argument('-lemb', '--load_emb',
                        help='load saved embeddings (default: False)')
    parser.add_argument('-lexp', '--load_exp',
                        help='load saved experiment results (default: False)')
    parser.add_argument('-rounds', '--rounds',
                        help='number of rounds (default: 5)')
    parser.add_argument('-plot', '--plot',
                        help='plot the results (default: False)')
    parser.add_argument('-saveMAP', '--save_MAP',
                        help='save MAP in a latex table (default: False)')

    params = json.load(open('gem/experiments/config/params2.conf', 'r'))
    args = vars(parser.parse_args())

    for k, v in args.items():
        if v is not None:
            params[k] = v

    params["experiments"] = params["experiments"].split(',')
    params["data_sets"] = params["data_sets"].split(',')
    params["rounds"] = int(params["rounds"])
    params["n_sample_nodes"] = int(params["n_sample_nodes"])
    params["is_undirected"] = bool(int(params["is_undirected"]))

    params["dimensions"] = params["dimensions"].split(',')
    if "nc_test_ratio_arr" in params:
        params["nc_test_ratio_arr"] = params["nc_test_ratio_arr"].split(',')
        params["nc_test_ratio_arr"] = \
            [float(ratio) for ratio in params["nc_test_ratio_arr"]]
    
    for data_set in params["data_sets"]:
        call_exps(params, data_set)
