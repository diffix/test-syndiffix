import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cats = ['a', 'b', 'c', 'd', 'f', 'g']
rgbColors = sns.color_palette('husl', len(cats))

colors = {}
for i in range(len(cats)):
    colors[cats[i]] = rgbColors[i]

print(colors)

moreCats = cats + ['x','y','z']

# Generate a list of 100 random categories
#cat_column = np.random.choice(moreCats, 100)
cat_column = np.random.choice(cats, 100)

# Generate a list of 100 random values
val_column = np.random.rand(100)

# Create a DataFrame
df = pd.DataFrame({
    'cat': cat_column,
    'val': val_column
})

# Print the DataFrame
print(df)

#sns.boxplot(x='val', y='cat', data=df, palette=colors)
sns.boxplot(x=df['val'], y=df['cat'],  palette=colors)
plt.savefig("out.png")
plt.close()

#sns.boxplot(x='val', y='cat', data=df, palette=None)
sns.boxplot(x=df['val'], y=df['cat'],  palette=None)
plt.savefig("out2.png")
plt.close()

df1 = df.query("cat == 'a' or cat == 'b' or cat == 'c'")
#sns.boxplot(x='val', y='cat', data=df1, palette=colors)
sns.boxplot(x=df1['val'], y=df1['cat'],  palette=colors)
plt.savefig("out1.png")
plt.close()