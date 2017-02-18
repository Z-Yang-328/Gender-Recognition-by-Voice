#This program is to visualize some features of original dataset

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

#First to see our dataset
voice = pd.read_csv("voice.csv")
print("Sample dataset")
print(50 * '-')
print(voice.head())

#Scatter plot of given features
sns.FacetGrid(voice, hue="label", size=5)\
   .map(plt.scatter, "meanfun", "meanfreq")\
   .add_legend()
plt.show()

#Boxplot
sns.boxplot(x="label",y="meanfun",data=voice)
plt.show()

#Boxplot
sns.boxplot(x="label",y="meanfun",data=voice)
plt.show()

#Distribution of male and female(every feature)
sns.FacetGrid(voice, hue="label", size=6) \
   .map(sns.kdeplot, "meanfun") \
   .add_legend()
plt.show()

#Radviz circle plot
from pandas.tools.plotting import radviz
radviz(voice, "label")
plt.show()
