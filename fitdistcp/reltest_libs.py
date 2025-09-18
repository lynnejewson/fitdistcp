import numpy as np
import matplotlib.pyplot as plt


def format_data_a(actual_p_ml, actual_p_cp, p):
    x = []
    y_ml = []
    y_cp = []
    for i in range(len(p)):
        x.append(p[len(p)-1-i])
        y_ml.append(actual_p_ml[len(p)-1-i])
        y_cp.append(actual_p_cp[len(p)-1-i])
    x_range = (x[0], x[-1])
    y_range = (y_ml[0], y_ml[-1])
    return {'xlabel': 'Nominal Probability',  'ylabel': 'PCP', 'x':x, 'y_ml':y_ml, 'y_cp':y_cp, 
            'title': '(a) PCP vs NP', 'benchmark':{'x': x_range, 'y': x_range}, 'limits': {'x': x_range, 'y': y_range}}


def format_data_b(actual_p_ml, actual_p_cp, p):
    x = []
    y_ml = []
    y_cp = []
    for p_val in 0.0001 * np.asarray(range(1000,-1,-1)):
        x.append(p_val)
        y_ml.append(actual_p_ml[int(p_val*10000)])
        y_cp.append(actual_p_cp[int(p_val*10000)])
    x_range = (x[0], x[-1])
    y_range = x_range
    return {'xlabel': 'Nominal Probability',  'ylabel': 'PCP', 'x':x, 'y_ml':y_ml, 'y_cp':y_cp, 
            'title': '(b) PCP vs NP (tail)', 'benchmark':{'x': x_range, 'y': x_range}, 'limits': {'x': x_range, 'y': y_range}}


def format_data_d(actual_p_ml, actual_p_cp, p):
    x = []
    y_ml = []
    y_cp = []
    cut_off_index = 200 # since graph peaks sharply, can ignore first (p smallest) n points
    for i in range(len(p) - cut_off_index):
        i_rev = len(p) - 1 - i
        x.append(p[i_rev])
        y_ml.append(actual_p_ml[i_rev]/p[i_rev])
        y_cp.append(actual_p_cp[i_rev]/p[i_rev])
    x_range = (x[0], x[-1])
    y_range = (y_ml[0], y_ml[-1])
    return {'xlabel': 'Nominal Probability',  'ylabel': 'Prob Ratio', 'x':x, 'y_ml':y_ml, 'y_cp':y_cp, 
            'title': '(d) PCP/NP vs NP', 'benchmark':{'x': x_range, 'y': (1,1)}, 'limits': {'x': x_range, 'y': y_range}}


def format_data_h(actual_p_ml, actual_p_cp, p):
    x = []
    y_ml = []
    y_cp = []
    cut_off_index = 30
    for i in range(len(p) - cut_off_index):
        i_rev = len(p) - 1 - i
        x.append(1/p[i_rev])
        y_ml.append(1/actual_p_ml[i_rev])
        y_cp.append(1/actual_p_cp[i_rev])
    x_range = (x[0], x[-1])
    return {'xlabel': '1/Nominal Probability',  'ylabel': '1/PCP', 'x':x, 'y_ml':y_ml, 'y_cp':y_cp, 
            'title': '(h) 1/PCP vs 1/NP', 'benchmark':{'x': x_range, 'y': x_range}, 'limits': {'x': x_range, 'y': x_range}}


def format_data_i(actual_p_ml, actual_p_cp, p):
    x = []
    y_ml = []
    y_cp = []
    cut_off_index = 40  # cut off both the highest and lowest p values
    for i in range(len(p) - 2*cut_off_index):
        i_rev = len(p)-1-i-cut_off_index
        np = p[i_rev]
        pcp_ml = actual_p_ml[i_rev]
        pcp_cp = actual_p_cp[i_rev]
        x.append(1/np - 1/(1-np))
        y_ml.append(1/pcp_ml - 1/(1-pcp_ml))
        y_cp.append(1/pcp_cp - 1/(1-pcp_cp))
    x_range = (x[0], x[-1])
    return {'xlabel': '1/NP-1/(1-NP)',  'ylabel': '1/PCP-1/(1-PCP)', 'x':x, 'y_ml':y_ml, 'y_cp': y_cp,
            'title': '(h) PCP vs NP', 'benchmark':{'x': x_range, 'y': x_range}, 'limits': {'x': x_range, 'y': x_range}}



###################################################################################


formats = {
    'unformatted': format_data_a,
    'b': format_data_b,
    'd': format_data_d,
    'tail': format_data_h,
    'i': format_data_i
}



def single_plot(data, ax):
    # data is the dictionary returned by my format_data routines
    ax.plot(data['x'], data['y_ml'],  label='ML', color='red', linewidth=1)
    ax.plot(data['x'], data['y_cp'],  label='CP', color='blue')
    ax.plot(data['benchmark']['x'], data['benchmark']['y'], color='black', label='Benchmark')
    ax.set_xlim(data['limits']['x'])
    ax.set_ylim(data['limits']['y'])
    ax.set_xlabel(data['xlabel'])
    ax.set_ylabel(data['ylabel'])
    ax.set_title(data['title'])
    ax.legend()                         # crowds the graphs somewhat


def plot_all(result):
    '''
    Passed output from a reltest, makes a plot with 5 formats comparing the ML and CP methods (using matplotlib).

    Parameters
    ----------
    result: Dict
        output from reltests (with keys actual_p_ml, actaul_p_cp, and p)
    '''
    actual_p_ml = result['actual_p_ml']
    actual_p_cp = result['actual_p_cp']
    p = result['p']

    data = [
        format_data_a(actual_p_ml, actual_p_cp, p), 
        format_data_b(actual_p_ml, actual_p_cp, p), 
        format_data_d(actual_p_ml, actual_p_cp, p), 
        format_data_h(actual_p_ml, actual_p_cp, p), 
        format_data_i(actual_p_ml, actual_p_cp, p)
        ]
    fig, axs = plt.subplots(3,2)
    single_plot(data[0], axs[0,0])
    single_plot(data[1], axs[0,1])
    single_plot(data[2], axs[1,0])
    single_plot(data[3], axs[1,1])
    single_plot(data[4], axs[2,0])
    fig.delaxes(axs[2,1])
    fig.tight_layout()
    plt.show()



def plot(result, key='tail'):
    '''
    Passed output from a reltest, makes a plot comparing the ML and CP methods (using matplotlib).
    Possible keys: 'unformatted', 'b', 'd', 'tail', 'i', 'all'.
    'tail' demonstrates tail probabilities the best.

    Parameters
    ----------
    result: Dict
        output from reltests (with keys actual_p_ml, actaul_p_cp, and p)
    key: str
        Determines how the reltest output should be formatted.
    '''

    if key not in ['unformatted', 'b', 'd', 'i', 'tail', 'all']:
        raise Exception('invalid reltest plot_option: must be one of unformatted, b, d, i, tail, all')

    if key=='all':
        plot_all(result)
    else:
        data = formats[key](
            result['actual_p_ml'], 
            result['actual_p_cp'], 
            result['p'])
        fig, axs = plt.subplots(1, 1)
        single_plot(data, axs)
        plt.show()