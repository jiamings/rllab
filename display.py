import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from run import get_archs


def moving_average(v, window=100):
    ret = np.cumsum(v, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window-1:] / window


def num_neurons(neurons=[64]):
    return lambda x: sum(x) in neurons


def num_layers(layers=[1]):
    return lambda x: len(x) in layers


def layer_neurons(neurons=[64]):
    return lambda x: x[0] in neurons


def get_pyfig(exp, cond=lambda x: True, top=0):
    archs = get_archs()
    path = 'data/local/{}/'.format(exp)
    d = dict()
    for arch in archs:
        ac = [int(a) for a in arch.split('-')]
        if not cond(ac):
            continue
        d[arch] = pd.read_csv(path + arch + '/progress.csv')['AverageReturn'].values

    if top > 0:
        c = [v_[-1] for v_ in d.values()]
        c = sorted(c)[-top]
        for k, v in d.items():
            if v[-1] < c:
                del d[k]

    traces = dict()
    for k, v in d.items():
        v = moving_average(v, v.shape[0] / 50)
        traces[k] = go.Scatter(
            x = np.arange(1, v.shape[0], 1),
            y = v,
            mode = 'lines',
            name = k
        )
    layout = dict(
        title=exp,
        xaxis=dict(title='Iterations'),
        yaxis=dict(title='Average Reward')
    )
    data = [v for k, v in traces.items()]
    return dict(data=data, layout=layout)
