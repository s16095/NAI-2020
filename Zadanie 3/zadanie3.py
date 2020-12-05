# Zadanie 3 - sterownik poduszki powietrzznej w samochodzie
# Autorzy:
# Wojciech Iracki, Adrian Wojewoda

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# New Antecedent/Consequent objects hold universe variables and membership
# functions
speed = ctrl.Antecedent(np.arange(0, 100, 10), 'speed')
collision_power = ctrl.Antecedent(np.arange(0, 10, 1), 'collision_power')
airbag_power = ctrl.Consequent(np.arange(0, 100, 10), 'airbag_power')
# Auto-membership function population is possible with .automf(3, 5, or 7)
speed.automf(3)
collision_power.automf(3)
# Custom membership functions can be built interactively with a familiar,
# Pythonic API
airbag_power['low'] = fuzz.trimf(airbag_power.universe, [0, 0, 25])
airbag_power['medium'] = fuzz.trimf(airbag_power.universe, [0, 25, 50])
airbag_power['high'] = fuzz.trimf(airbag_power.universe, [50, 75, 100])
"""
To help understand what the membership looks like, use the ``view`` methods.
These return the matplotlib `Figure` and `Axis` objects. They are persistent
as written in Jupyter notebooks; other environments may require a `plt.show()`
command after each `.view()`.
"""
# You can see how these look with .view()
speed['average'].view()
"""
.. image:: PLOT2RST.current_figure
"""
collision_power.view()
"""
.. image:: PLOT2RST.current_figure
"""
airbag_power.view()
"""
.. image:: PLOT2RST.current_figure
Fuzzy rules
-----------
Now, to make these triangles useful, we define the *fuzzy relationship*
between input and output variables. For the purposes of our example, consider
three simple rules:
1. If the food is poor OR the service is poor, then the tip will be low
2. If the service is average, then the tip will be medium
3. If the food is good OR the service is good, then the tip will be high.
Most people would agree on these rules, but the rules are fuzzy. Mapping the
imprecise rules into a defined, actionable tip is a challenge. This is the
kind of task at which fuzzy logic excels.
"""
rule1 = ctrl.Rule(speed['poor'] | collision_power['poor'], airbag_power['low'])
rule2 = ctrl.Rule(collision_power['average'], airbag_power['medium'])
rule3 = ctrl.Rule(speed['good'] | collision_power['good'], airbag_power['high'])
rule1.view()
"""
.. image:: PLOT2RST.current_figure
Control System Creation and Simulation
---------------------------------------
Now that we have our rules defined, we can simply create a control system
via:
"""
airbag_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
"""
In order to simulate this control system, we will create a
``ControlSystemSimulation``.  Think of this object representing our controller
applied to a specific set of circumstances.  For tipping, this might be tipping
Sharon at the local brew-pub.  We would create another
``ControlSystemSimulation`` when we're trying to apply our ``tipping_ctrl``
for Travis at the cafe because the inputs would be different.
"""
airbag = ctrl.ControlSystemSimulation(airbag_ctrl)
"""
We can now simulate our control system by simply specifying the inputs
and calling the ``compute`` method.  Suppose we rated the quality 6.5 out of 10
and the service 9.8 of 10.
"""
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
airbag.input['speed'] = 90
airbag.input['collision_power'] = 1
# Crunch the numbers
airbag.compute()
"""
Once computed, we can view the result as well as visualize it.
"""
print(airbag.output['airbag_power'])
collision_power.view(sim=airbag)
