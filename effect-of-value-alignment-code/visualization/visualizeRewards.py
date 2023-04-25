import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, NumericInput
from bokeh.plotting import figure
from utils import get_expected_reward_robot

# Set up data
stepsize = 0.01
num_weights = int(1/stepsize) + 1
wh_rob = np.linspace(0.0, 1.0, num_weights)
wc_rob = 1.0 - wh_rob

r0_rob, r1_rob = get_expected_reward_robot(wh_rob, 0.5, 10.0, 10.0, 0.2, 0.8, 0.5)
source0 = ColumnDataSource(data=dict(x=wh_rob, y=r0_rob))
source1 = ColumnDataSource(data=dict(x=wh_rob, y=r1_rob))

# Set up plots
plot = figure(height=400, width=400, title='Expected Rewards for the robot',
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0.0,1.0])

plot.line('x', 'y', source=source0, line_width=3, line_alpha=0.6, line_color='black', legend_label='NOT USE')
plot.line('x', 'y', source=source1, line_width=3, line_alpha=0.6, line_color='blue', legend_label='USE')


# Set up widgets
# Sliders
wh_hat_slider = Slider(title="wh_hat", value=0.5, start=0.0, end=1.0, step=0.1)
# wh_rob_slider = Slider(title='wh_rob', value=0.7, start=0.0, end=1.0, step=0.1)
d_k_slider = Slider(title='d_k', value=0.5, start=0.0, end=1.0, step=0.1)
trust_slider = Slider(title='trust', value=0.5, start=0.0, end=1.0, step=0.1)

# Numeric inputs
h_inp = NumericInput(title='h', value=10.0)
c_inp = NumericInput(title='c', value=10.0)
kappa_inp = NumericInput(title='kappa', value=0.2)

# Set up callbacks

def update_data(attr_name, old, new):
    # Estimated weights
    wh_hat = wh_hat_slider.value
        
    # Threat level
    d_k = d_k_slider.value
    
    # Trust level
    trust = trust_slider.value
    
    # Health loss cost
    h = -h_inp.value
    
    # Time loss cost
    c = -c_inp.value
    
    # Rationality coefficient
    kappa = kappa_inp.value
    
    # Generate the new curve    
    wh_rob = np.linspace(0.0, 1.0, num_weights)
    r0_rob, r1_rob = get_expected_reward_robot(wh_rob, wh_hat, h, c, kappa, trust, d_k)
    
    source0.data = dict(x=wh_rob,y=r0_rob)
    source1.data = dict(x=wh_rob,y=r1_rob)

for w in [wh_hat_slider, d_k_slider, trust_slider, h_inp, c_inp, kappa_inp]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(h_inp, c_inp, kappa_inp, wh_hat_slider, d_k_slider, trust_slider)

curdoc().add_root(row(inputs, plot, width=800))

