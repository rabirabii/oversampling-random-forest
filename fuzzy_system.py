import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
def create_fuzzy_system():
    glucose = ctrl.Antecedent(np.arange(0,200,1), 'glucose')
    bmi = ctrl.Antecedent(np.arange(0,50,1), 'bmi')
    age = ctrl.Antecedent(np.arange(0,100,1), 'age')

    # Output 
    risk = ctrl.Consequent(np.arange(0,101,1), 'risk')

    # Input
 
    glucose['very low'] = fuzz.trimf(glucose.universe, [0, 0, 70])
    glucose['low'] = fuzz.trimf(glucose.universe, [0, 70, 100])
    glucose['normal'] = fuzz.trimf(glucose.universe, [70, 100, 125])
    glucose['high'] = fuzz.trimf(glucose.universe, [100, 125, 200])
    glucose['very high'] = fuzz.trimf(glucose.universe, [125, 200, 200])

    # Fuzzy sets for BMI
    bmi['underweight'] = fuzz.trimf(bmi.universe, [0, 0, 18.5])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 23, 25])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [25, 27.5, 30])
    bmi['obese'] = fuzz.trimf(bmi.universe, [30, 35, 40])
    bmi['extremely obese'] = fuzz.trimf(bmi.universe, [35, 40, 50])

    # Fuzzy sets for age
    age['young'] = fuzz.trimf(age.universe, [0, 0, 35])
    age['middle'] = fuzz.trimf(age.universe, [30, 45, 60])
    age['old'] = fuzz.trimf(age.universe, [55, 70, 85])
    age['very old'] = fuzz.trimf(age.universe, [80, 100, 100])

    # Fuzzy sets for risk
    risk['very low'] = fuzz.trimf(risk.universe, [0, 0, 25])
    risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
    risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
    risk['high'] = fuzz.trimf(risk.universe, [50, 75, 100])
    risk['very high'] = fuzz.trimf(risk.universe, [75, 100, 100])


     # Fuzzy rules
    rule1 = ctrl.Rule(glucose['very high'] & bmi['obese'] & age['old'], risk['very high'])
    rule2 = ctrl.Rule(glucose['normal'] & bmi['normal'] & age['young'], risk['very low'])
    rule3 = ctrl.Rule(glucose['high'] & bmi['overweight'] & age['middle'], risk['medium'])
    rule4 = ctrl.Rule(glucose['high'] & bmi['obese'], risk['high'])
    rule5 = ctrl.Rule(glucose['very high'] & age['old'], risk['very high'])
    rule6 = ctrl.Rule(bmi['extremely obese'] & age['middle'], risk['high'])
    rule7 = ctrl.Rule(glucose['low'] & bmi['normal'] & age['young'], risk['low'])
    rule8 = ctrl.Rule(glucose['normal'] & bmi['overweight'] & age['middle'], risk['medium'])
    rule9 = ctrl.Rule(glucose['high'] & age['very old'], risk['very high'])
    rule10 = ctrl.Rule(bmi['underweight'] & age['old'], risk['medium'])
    rule11 = ctrl.Rule(glucose['very low'] & bmi['normal'], risk['low'])
    rule12 = ctrl.Rule(glucose['normal'] & bmi['obese'] & age['young'], risk['medium'])
    rule13 = ctrl.Rule(glucose['high'] & bmi['normal'] & age['middle'], risk['medium'])
    rule14 = ctrl.Rule(bmi['extremely obese'] & age['old'], risk['very high'])
    rule15 = ctrl.Rule(glucose['very high'] & bmi['normal'] & age['young'], risk['high'])

    # Control System
    risk_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15])
    risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

    return risk_simulation