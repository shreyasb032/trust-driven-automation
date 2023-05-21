import numpy as np
from matplotlib import pyplot as plt
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, NumericInput
from bokeh.plotting import figure
from scipy.stats import beta


def update_data(attrname, old, new):
    a = aInput.value
    b = a / 9

    bInput.value = b
    x = np.linspace(1e-6, 1.0-1e-6, N)
    y = beta.pdf(x, a, b)
    source.data = dict(x=x, y=y)

fig, ax = plt.subplots()

N = 500
a = 28
b = 4
x = np.linspace(1e-6, 1.0-1e-6, N)
y = beta.pdf(x, a, b)
source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(height=800, width=800, title="Beta Distribution",
            tools="crosshair,pan,reset,save,wheel_zoom",
            x_range=[0, 1]) #, y_range=[-0.01, 2.5])

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

aInput = NumericInput(title = 'alpha', value=28)
bInput = NumericInput(title = 'beta', value=4)

aInput.on_change('value', update_data)

inputs = column(aInput, bInput)

curdoc().add_root(row(inputs, plot, width=1200))
