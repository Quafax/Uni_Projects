#Here will be the homework 1
#Task 1.1 Markov Decision Process of Monty Hall Problem
import numpy as np
P_stay= np.array([
    #Si RevB RevC RevB_PrA RevB_PrC RevC_PrA RevC_PrB
    [0,1/2,1/2,0,0,0,0],
    [0,0,0,1/3,2/3,0,0],
    [0,0,0,0,0,1/3,2/3],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0]
    ]
)
#Smarter would be np.zeros but for clearity its like this

#P_switch has to be the same bec only rewards are changed
P_switch= np.array([
    #Si RevB RevC RevB_PrA RevB_PrC RevC_PrA RevC_PrB
    [0,1/2,1/2,0,0,0,0],
    [0,0,0,1/3,2/3,0,0],
    [0,0,0,0,0,1/3,2/3],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0]
    ]
)

#Task 1.2
R_stay = [0,0,0,1,0,1,0]
R_switch = [0,0,0,0,1,0,1]

#Task1.3
def values(gamma,P,R):
    I = np.eye(7)
    v= np.linalg.inv(I - gamma * P)@R
    #korrekt ist Zeile = Ausgangszustand und Spalte = nächster Zustand. sonst matrix transponieren
    return v
#Values of Stay is: [0.33333333 0.33333333 0.33333333 1.     0.   1.      0.  ]
#Values of switch is: [0.66666667 0.66666667 0.66666667 0.   1.  0.       1. ]
#These make sense bec its smarter to switch and gives a higher chance for winning

if __name__ == '__main__':
    gamma=1
    v_stay = values(gamma,P_stay,R_stay)
    v_switch = values(gamma,P_switch,R_switch)
    print("Values of switch is:", v_switch)
