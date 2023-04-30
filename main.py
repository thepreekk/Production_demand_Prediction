from Working_Model import *
import numpy as np

Total_price = float(input("What is TOTAL PRICE of the Product ?"))
Base_price = float(input("What is the BASE PRICE of the Product ?"))

features = np.array([[Total_price, Base_price]])
prediction = model.predict(features)
print("\n")
print("Quantity can be demanded based on input Price = ", prediction)
