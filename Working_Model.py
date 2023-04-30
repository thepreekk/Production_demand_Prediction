import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("Productdata.csv")
data_head = data.head()
# print(data_head)

# Checking whether the data set contains any null values or not.

check_null = data.isnull().sum()
# print(check_null)

# removing the row which contains the missing values
data = data.dropna()

# showing a how a demand for product varies with the change in price.
fig = px.scatter(data, x="Units Sold", y="Total Price", size='Units Sold')
fig.show()

# Relations between attributes of datasets
print("showing Co-relation between Dataset :-")
co_relation = data.corr()
# print(co_relation)

correlations = data.corr(method='pearson')
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
# plt.savefig("co-relation heatmap")
# plt.show()

# Product demand Prediction Model
""" Training a machine learning model to predict the demand for the product at different prices. 
Let's choose the Total Price and the Base Price column as the features and the Units Sold column 
as labels for the model:"""

x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(xtrain.values, ytrain)

