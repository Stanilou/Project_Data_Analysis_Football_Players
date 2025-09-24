# Import required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read and prepare the data
data = pd.read_csv("cleaned_football_dataset.csv")

# Select numerical columns for PCA (excluding ID, Player, Nation, Pos, Squad, Comp)
numerical_columns = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A']
X = data[numerical_columns]

# Standardize the features
X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numerical_columns)

# Perform PCA
n_components = 6  # We'll keep 6 components to analyze variance explained
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(X_scaled)

# Create DataFrame with eigenvalues summary
eig = pd.DataFrame({
    "Dimension": [f"Dim{x+1}" for x in range(n_components)],
    "Valeur propre": pca.explained_variance_,
    "% valeur propre": np.round(pca.explained_variance_ratio_ * 100, 2),
    "% cum. val. prop.": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
})

print("Eigenvalues Summary:")
print(eig)

# Plot scree plot (variance explained by each component)
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.show()

# Function for biplot
def biplot(score, coeff, coeff_labels=None):
    plt.figure(figsize=(10, 8))
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    
    plt.scatter(xs, ys, c='b', alpha=0.5)
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if coeff_labels is not None:
            plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, coeff_labels[i], color='g')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()
    
# Create biplot with variable names
biplot(pca_result[:,:2], 
       pca.components_.T[:,:2],
       coeff_labels=numerical_columns)
plt.title('PCA Biplot')
plt.show()