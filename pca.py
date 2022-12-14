import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data = np.array([[1000 ,500],[2000, 800],[3000 ,1100],[4000 ,1500],[5000 ,1800],[8000,1900]])
df = pd.DataFrame(data,columns = ['Salary','Expense'])

fig = px.scatter(df, x="Salary", y="Expense", trendline="ols")
fig.show()

df

"""<p>Scaling the data</p>"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
scaled

"""<p>Finding the covariance matrix</p>"""

cov_mat = np.cov(scaled[:,0], scaled[:,1])
cov_mat

fig = px.imshow(cov_mat, text_auto=True)
fig.show()

"""So, PCA is a method that:

    Measures how each variable is associated with one another using a Covariance matrix
    Understands the directions of the spread of our data using Eigenvectors
    Brings out the relative importance of these directions using Eigenvalues

<p>Finding the eigen values and eigen vectors</p>
"""

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


print(f'Eigen Values: \n{eig_vals}')
print(f'\nEigen Vectors: \n{eig_vecs}')

"""<p>
In order to decide which eigenvector(s) can dropped without losing too much information for the construction of lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped.
</p>
"""

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

"""<p>Explained Variance After sorting the eigenpairs, the next question is "how many principal components are we going to choose for our new feature subspace?" A useful measure is the so-called "explained variance," which can be calculated from the eigenvalues. The explained variance tells us how much information (variance) can be attributed to each of the principal components.</p>"""

# tot = sum(eig_vals)
# var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
# var_exp

sum = 0

for i in eig_vals:
  sum += i

var_exp = list()

for i in eig_vals:
  x = i / sum
  x *= 100

  var_exp.append(x)

print(var_exp)

len(var_exp)

plt.style.use("dark_background")
plt.figure(figsize=(4,6))
sns.barplot(x=[1,2],y=var_exp)

eig_pairs

matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1)))
print('Matrix W:\n', matrix_w)

eig_vecs[0].T

final_data = np.dot(scaled, np.array(matrix_w))
print(final_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(scaled)
print("Varaince explained by principal component is \n", pca.explained_variance_ratio_)
print("Final output after PCA \n",pca.transform(scaled)[:,0])

"""<h2>Example 2</h2>"""

data = np.array([[90,90,25,95,100],[80,95,40,85,77],[50,30,95,87,27],[27,37,25,68,25],[25,41,88,63,36],[41,42,45,61,78]])
df = pd.DataFrame(data,columns = ['Maths','Science','Social','English','Lang-II'])

df

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
scaled

mean_vec = np.mean(scaled, axis=0)
cov_mat = (scaled - mean_vec).T.dot((scaled - mean_vec)) / (scaled.shape[0]-1)
cov_mat

np.cov(scaled.T)

fig = px.imshow(cov_mat, text_auto=True)
fig.show()

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


print(f'Eigen Values: \n{eig_vals}')
print(f'\nEigen Vectors: \n{eig_vecs}')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
var_exp

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

eig_pairs

n = len(eig_pairs)
matrix_w = np.hstack((eig_pairs[0][1].reshape(n,1), 
                      eig_pairs[1][1].reshape(n,1)
                    ))
print('Matrix W:\n', matrix_w)

final_data = np.dot(scaled, np.array(matrix_w))
print(final_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled)
print("Varaince explained by principal component is \n", pca.explained_variance_ratio_)
print("Final output after PCA \n",pca.transform(scaled))

fig = px.imshow(np.corrcoef(final_data.T), text_auto=True)
fig.show()

fig = px.imshow(np.corrcoef(scaled.T), text_auto=True)
fig.show()

np.cov(scaled.T)

np.cov(final_data.T)

"""<h3>Iris Dataset</h3>"""

import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()

import plotly.express as px
from sklearn.decomposition import PCA

df = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

pca = PCA()
components = pca.fit_transform(df[features])
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=df["species"]
)
fig.update_traces(diagonal_visible=False)
fig.show()

"""<h2>MNIST DATASET</h2>"""

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/train (1).csv')

df.head()

df.info()

df.describe()

y = df.pop('label')

df.columns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df.cov()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
scaled.shape

cov_mat = np.matmul(scaled.T, scaled)
cov_mat.shape

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


print(f'Eigen Values: \n{eig_vals}')
print(f'\nEigen Vectors: \n{eig_vecs}')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
var_exp

with plt.style.context('dark_background'):
    plt.figure(figsize=(20,10))

    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

n = len(eig_pairs)
matrix_w = np.hstack((eig_pairs[0][1].reshape(n,1), 
                      eig_pairs[1][1].reshape(n,1)
                    ))
print('Matrix W:\n', matrix_w)

final_data = np.dot(scaled, np.array(matrix_w))
print(final_data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled)
print("Varaince explained by principal component is \n", pca.explained_variance_ratio_)
print("Final output after PCA \n",pca.transform(scaled))

dataFrame = pd.DataFrame(final_data, columns = ['pca_1', 'pca_2'])
dataFrame.cov()

dataFrame['label'] = y
dataFrame.head()

sns.FacetGrid(dataFrame, hue = 'label', size = 8).map(sns.scatterplot, 'pca_1', 'pca_2').add_legend()
plt.show()

fig = px.scatter(dataFrame, x="pca_1", y="pca_2", color="label", hover_data=['label'])

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)

fig.show()

