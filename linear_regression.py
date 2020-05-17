import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# Load dataset

data = pd.read_csv('D:/Linear Regression/dataset.csv')

# Show first five rows of data
print("First five examples are:","\n")
print(data.head(), '\n')

# Dimensions of dataset
print("The dimensions of dataset are: ", data.shape, '\n')


# Plot data points
data.plot(kind='scatter', x='Population', y='Profit')
plt.title("Dataset")
plt.xlabel("Population in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

# Boxplot
data.plot(kind='box')
plt.title("Boxplot")
plt.show()

#coefficient of correlation
print("Correlation Matrix is: ", '\n')
correlation = data.corr()
print(correlation, '\n')

#change to dataframe variables
population = pd.DataFrame(data['Population'])
profit = pd.DataFrame(data['Profit'])


# Build Linear Regression Model

lm = linear_model.LinearRegression()
model = lm.fit(population, profit)

print("Model Coefficient is: ", model.coef_, '\n')

print("Model Intercept is: ", model.intercept_, '\n')

score = model.score(population, profit)
print("The accuracy of model is: ", score, '\n')

#evaluate model


# predict new profit value

# initialize empty list of test data
new_population_values = []


n=int(input("Enter the no. of test examples: "))

print()
print("Enter the new population values: ")
for i in range(0, n):
	test_data = float(input())
	new_population_values.append(test_data)

#new_population_values = (new_population_values)
new_population_values = pd.DataFrame(new_population_values)
predicted_profit = model.predict(new_population_values)
predicted_profit = pd.DataFrame(predicted_profit)
df = pd.concat([new_population_values, predicted_profit], axis=1, keys=['Population in 10,000s', 'Profit in $10,000s'])
print('\n', df)


# Visualize the result

data.plot(kind='scatter', x='Population', y='Profit', color='red')

#Plotting the regression line
plt.plot(population, model.predict(population), color='blue', linewidth=1)

#Plotting the predicted value
plt.scatter(new_population_values, predicted_profit, color='black')
plt.title("Linear Regression Model")
plt.xlabel("Population in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

