import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

# Read csv file and print the column names
csv_doc = pd.read_csv('lmdata.csv')
print()
print(csv_doc.columns)
print('\n')

# Convert file into a dataframe
df = pd.DataFrame(csv_doc)
print(df.to_markdown(tablefmt="grid"))
print('\n')

# Generate and print the descriptive statistics of the dataframe
print(csv_doc.describe().to_markdown(tablefmt="grid"))
print('\n')


# Create a lineplot
def lineplot():
    sns.lineplot(x='latitude', y='mortality', data=df, palette='PiYG')
    plt.title('Mortality to Latitude Lineplot')
    plt.show()


lineplot()


# Create a titled scatterplot
def scatterplot():
    sns.scatterplot(x='latitude', y='mortality', data=df)
    plt.title('Mortality to Latitude Scatterplot')
    plt.show()


scatterplot()


# Create a mortality boxplot
def mortality_boxplot():
    sns.boxplot(x='mortality', data=df, color='Green')
    plt.title('Mortality Boxplot')
    plt.show()


mortality_boxplot()


# Create a pearson correlation test for the variables
def correlation_test():
    corr = df.corr(method='pearson')
    print(corr.to_markdown(tablefmt='grid'))
    print('\n')


correlation_test()


# Create a pairplot for the data
def pairplot():
    sns.pairplot(df)
    plt.show()
    print('\n')


pairplot()


# Create a regplot with a 95% confidence
def regplot():
    sns.regplot(x='mortality', y='latitude', data=df, color='g', marker='x', ci=95)
    plt.title('Mortality to Latitude Pairplot')
    plt.show()


regplot()

# Create the regression model using ols() and print the summary
model = smf.ols('mortality ~ latitude', data=df)
model = model.fit()
print(model.summary())
