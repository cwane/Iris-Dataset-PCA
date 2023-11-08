
#Importing all necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

df= pd.DataFrame(data=iris.data, columns=iris.feature_names)


df.info()


print(df)


df.head()


df['target'] = iris.target


df.head()


df.head(10)

#mapping the target with species column

df['target']=pd.Series(iris.target)
df.head(150)


X = df.iloc[:,0:4].values
#print(X)

print(np.mean(X))
print(np.std(X))


mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)

print("Mean of X:", mean_X)
print("Standard deviation of X:", std_X):


X_std = StandardScaler().fit_transform(X)

print(np.mean(X_std))
print(np.std(X_std))


print(X_std.shape)


cov_mat = np.cov(np.transpose(X_std))
print(cov_mat)
cov_mat.shape

eigen_value,eigen_vector = np.linalg.eig(cov_mat)
print(eigen_value)
print(eigen_vector)
eigen_value.shape


eigen_vector.shape


#calculation of all the prportions
prop1 = eigen_value[0] / np.sum(eigen_value)
print(prop1*100)


prop2 = eigen_value[1] / np.sum(eigen_value)
print(prop2*100)

prop3 = eigen_value[2] / np.sum(eigen_value)
print(prop3*100)

prop4 = eigen_value[3] / np.sum(eigen_value)
print(prop4*100)

#scree plot of eigen values
eigenvalues = [eigen_value[0], eigen_value[1], eigen_value[2], eigen_value[3]]

# Calculate the total sum of eigenvalues (total variance)
total_variance = np.sum(eigenvalues)

# Calculate the proportion of variance explained by each principal component
explained_variances = [(eig_val / total_variance) * 100 for eig_val in eigenvalues]

# Plot the scree plot
plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, len(explained_variances) + 1), explained_variances, color='b', alpha=0.7)
plt.xticks(range(1, len(explained_variances) + 1))
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Scree Plot - Variance Explained by Principal Components')

# Add percentage labels on top of each bar
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{explained_variances[i]:.1f}%', 
             ha='center', va='bottom')
    roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

plt.show()

#selection of principle component
best_eigen_vector = np.transpose( eigen_vector[:,0:1])
selected_eigenvectors = np.transpose(eigen_vector[:, :2])
mat_trans = np.transpose(X_std)

new_data = np.dot(selected_eigenvectors,mat_trans)
new_data.shape


selected_eigenvectors.shape



print(new_data.shape)
print(y.shape)

#Plot for PCA of iris dataset by taking two principle components
# Define the species-color mapping
species_color_dict = {
    0: 'blue',
    1: 'orange',
    2: 'green'
}

# Convert target values to corresponding colors
colors = [species_color_dict[val] for val in y]

# Scatter plot with colors based on species
plt.scatter(new_data[0, :], new_data[1, :], c=colors)

plt.title('Iris Dataset Scatter Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Create a list of legend handles for species-color mapping
legend_labels = []
species_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
    species_labels.append(iris.target_names[species])

# Add the species legend with species names
species_legend = plt.legend(handles=legend_labels, labels=species_labels, loc='upper left')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
plt.gca().add_artist(species_legend)

#plt.grid(True)
plt.show()


#choosing 3 principle components
best_eigen_vector = np.transpose( eigen_vector[:,0:1])
selected_eigenvectors = np.transpose(eigen_vector[:, :3])
mat_trans = np.transpose(X_std)

new_data = np.dot(selected_eigenvectors,mat_trans)
new_data.shape



#Plot for PCA of iris dataset in 3d by taking three principle components
# Define the species-color mapping
species_color_dict = {
    0: 'blue',
    1: 'orange',
    2: 'green'
}

# Convert target values to corresponding colors
colors = [species_color_dict[val] for val in y]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors based on species
ax.scatter(new_data[0, :], new_data[1, :], new_data[2, :], c=colors)

ax.set_title('Iris Dataset Scatter Plot (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad =-0.8

# Create a list of legend handles for species-color mapping
legend_labels = []
species_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
    species_labels.append(iris.target_names[species])

# Add the species legend with species names
species_legend = plt.legend(handles=legend_labels, labels=species_labels, loc='upper left')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
ax.add_artist(species_legend)

plt.show()

#2d plot using scikit
# Fit PCA on the data
pca = PCA(n_components=2)
new_data = pca.fit_transform(X)

# Define the species-color mapping
species_color_dict = {
    0: 'blue',
    1: 'orange',
    2: 'green'
}

# Convert target values to corresponding colors
colors = [species_color_dict[val] for val in y]

# Scatter plot with colors based on species
plt.scatter(new_data[:, 0], new_data[:, 1], c=colors)

plt.title('PCA on Iris Dataset using Scikit ')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Create a list of legend handles for species-color mapping
legend_labels = []
species_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
    species_labels.append(iris.target_names[species])

# Add the species legend with species names
species_legend = plt.legend(handles=legend_labels, labels=species_labels, loc='lower right')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
plt.gca().add_artist(species_legend)

plt.show()

#3d plot using Scikit
# Fit PCA on the data
pca = PCA(n_components=3)
new_data = pca.fit_transform(X)

# Define the species-color mapping
species_color_dict = {
    0: 'blue',
    1: 'orange',
    2: 'green'
}

# Convert target values to corresponding colors
colors = [species_color_dict[val] for val in y]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors based on species
ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], c=colors)

ax.set_title('PCA on Iris Dataset using Scikit (3D)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad = -0.8

# Create a list of legend handles for species-color mapping
legend_labels = []
species_labels = []
for species, color in species_color_dict.items():
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
    species_labels.append(iris.target_names[species])

# Add the species legend with species names
species_legend = ax.legend(handles=legend_labels, labels=species_labels, loc='upper left')

# Add roll number legend
roll_numbers = ['Roll[35,42]']
roll_number_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8)
plt.legend(handles=[roll_number_legend], labels=roll_numbers, loc='upper right')

# Add the species legend back to the plot
ax.add_artist(species_legend)

plt.show()





