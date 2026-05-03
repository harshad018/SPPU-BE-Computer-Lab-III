import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def main():
    print("--- Fuzzy Logic Controller (Tipping Problem) ---")
    
    # New Antecedent/Consequent objects hold universe variables and membership functions
    quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
    service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
    tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    quality.automf(3)
    service.automf(3)

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
    tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
    tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

    # Rules
    rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
    rule2 = ctrl.Rule(service['average'], tip['medium'])
    rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

    # Control System Creation and Simulation
    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    # Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
    # Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
    tipping.input['quality'] = 6.5
    tipping.input['service'] = 9.8

    # Crunch the numbers
    tipping.compute()

    print(f"Input Quality: 6.5")
    print(f"Input Service: 9.8")
    print(f"Recommended Tip: {tipping.output['tip']:.2f}%")

if __name__ == "__main__":
    main()
