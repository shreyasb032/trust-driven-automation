import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, NumericInput
from bokeh.plotting import figure
from utils import *

# Set up data
stepsize = 0.01
num_weights = int(1/stepsize) + 1
threat_level = np.linspace(0.0, 1.0, num_weights)

r0_rob, r1_rob = get_expected_reward_robot(0.7, 0.5, -10.0, -10.0, 0.2, 0.8, threat_level, 0.0)
r0_hum, r1_hum = get_expected_reward_human(0.5, -10.0, -10.0, threat_level)

source0 = ColumnDataSource(data=dict(x=threat_level, y=r0_rob))
source1 = ColumnDataSource(data=dict(x=threat_level, y=r1_rob))
source2 = ColumnDataSource(data=dict(x=threat_level, y=r0_hum))
source3 = ColumnDataSource(data=dict(x=threat_level, y=r1_hum))

# Set up plots
plot = figure(height=800, width=800, title='Expected Rewards for the robot',
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0.0,1.0], y_range=[-16.0,0.0],
              x_axis_label='Threat Level',
              y_axis_label='Reward')

plot.line('x', 'y', source=source0, line_width=3, line_alpha=0.6, line_color='black', legend_label='NOT USE (robot)')
plot.line('x', 'y', source=source1, line_width=3, line_alpha=0.6, line_color='blue', legend_label='USE (robot)')
plot.line('x', 'y', source=source2, line_width=3, line_alpha=0.6, line_color='brown', legend_label='NOT USE (human)')
plot.line('x', 'y', source=source3, line_width=3, line_alpha=0.6, line_color='teal', legend_label='USE (human)')

# Set up widgets
# Sliders
wh_hat_slider = Slider(title="wh_hat", value=0.5, start=0.0, end=1.0, step=0.1)
wh_rob_slider = Slider(title='wh_rob', value=0.7, start=0.0, end=1.0, step=0.1)
wt_rob_slider = Slider(title='wt_rob', value=0.0, start=0.0, end=2.0, step=0.1)
trust_slider = Slider(title='trust', value=0.5, start=0.0, end=1.0, step=0.1)
wh_hum_slider = Slider(title="wh_hum", value=0.5, start=0.0, end=1.0, step=0.1)

# Numeric inputs
h_inp = NumericInput(title='h', value=10.0, mode='float')
c_inp = NumericInput(title='c', value=10.0, mode='float')
kappa_inp = Slider(title='kappa', value=0.2, start=0.0, end=2.0, step=0.1)

# Set up callbacks

def update_data(attr_name, old, new):
    # Estimated weights
    wh_hat = wh_hat_slider.value
    
    # Robot weights
    wh_rob = wh_rob_slider.value
    wt_rob = wt_rob_slider.value
    
    # Humans true weights
    wh_hum = wh_hum_slider.value
    
    # Threat level
    threat_level = np.linspace(0.0, 1.0, num_weights)
    
    # Trust level
    trust = trust_slider.value
    
    # Health loss cost
    h = -h_inp.value
    
    # Time loss cost
    c = -c_inp.value
    
    # Rationality coefficient
    kappa = kappa_inp.value
    
    # Generate the new curve
    r0_rob, r1_rob = get_expected_reward_robot(wh_rob, wh_hat, h, c, kappa, trust, threat_level, wt_rob)
    r0_hum, r1_hum = get_expected_reward_human(wh_hum, h, c, threat_level)
    
    source0.data = dict(x=threat_level, y=r0_rob)
    source1.data = dict(x=threat_level, y=r1_rob)
    source2.data = dict(x=threat_level, y=r0_hum)
    source3.data = dict(x=threat_level, y=r1_hum)

for w in [wh_hat_slider, wh_hum_slider, wh_rob_slider, wt_rob_slider, trust_slider, h_inp, c_inp, kappa_inp]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(h_inp, c_inp, kappa_inp, wh_hat_slider, wh_hum_slider, wh_rob_slider, wt_rob_slider, trust_slider)

curdoc().add_root(row(inputs, plot, width=1200))

