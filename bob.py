import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

data = pd.read_csv("cleaned_football_dataset.csv")

quantitative_vars = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A']
data_quantitative = data[quantitative_vars].dropna()

# Scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data_quantitative)

# ACP
pca = PCA()
pca_res = pca.fit_transform(x_scaled)

eig = pd.DataFrame({
    "Dimension": ["Dim" + str(x + 1) for x in range(len(pca.explained_variance_))],
    "Valeur propre": pca.explained_variance_,
    "% valeur propre": np.round(pca.explained_variance_ratio_ * 100, 2),
    "% cum. val. prop.": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
})

print("Tableau des valeurs propres:")
print(eig)

plt.figure(figsize=(10, 6))
y1 = list(pca.explained_variance_ratio_ * 100)
x1 = range(1, len(y1) + 1)
plt.bar(x1, y1)
plt.xlabel('Dimensions')
plt.ylabel('Pourcentage de variance expliquée (%)')
plt.title('Valeurs propres de l\'ACP')
plt.xticks(x1)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Cercle de corrélation
def biplot(score, coeff, labels=None, density=False):
    plt.figure(figsize=(10, 8))
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.8, head_width=0.02)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='r', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='r', ha='center', va='center', fontsize=9)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Dimension 1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100))
    plt.ylabel("Dimension 2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100))
    plt.grid(alpha=0.3)
    plt.title('Graphique des variables (ACP)')

coeff_labels = list(data_quantitative.columns)
biplot(pca_res[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=coeff_labels)
plt.show()

# Graphique des individus
pca_df = pd.DataFrame({
    "Dim1": pca_res[:, 0],
    "Dim2": pca_res[:, 1],
    "Player": data.loc[data_quantitative.index, "Player"],
    "Pos": data.loc[data_quantitative.index, "Pos"],
    "Comp": data.loc[data_quantitative.index, "Comp"]
})

plt.figure(figsize=(14, 10))
plt.scatter(pca_df["Dim1"], pca_df["Dim2"], alpha=0.6)

for i, (x, y, player) in enumerate(zip(pca_df["Dim1"], pca_df["Dim2"], pca_df["Player"])):
    plt.annotate(player, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=6, alpha=0.7)

plt.xlabel("Dimension 1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100))
plt.ylabel("Dimension 2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100))
plt.title("Graphique des individus")
plt.grid(alpha=0.3)
plt.show()

# 8. Graphique des individus coloré par position
plt.figure(figsize=(14, 10))

couleurs = {
    'FW': '#ff0000',
    'FW,MF': '#ff7700',
    'MF': '#ffaa00',
    'DF,MF': '#0077ff',
    'DF': '#0000ff'
}

positions = pca_df["Pos"].unique()

for pos in positions:
    mask = pca_df["Pos"] == pos
    plt.scatter(pca_df[mask]["Dim1"], pca_df[mask]["Dim2"], c=[couleurs[pos]], label=pos, alpha=0.7)

for i, (x, y, player) in enumerate(zip(pca_df["Dim1"], pca_df["Dim2"], pca_df["Player"])):
    plt.annotate(player, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=6, alpha=0.7)

plt.xlabel("Dimension 1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100))
plt.ylabel("Dimension 2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100))
plt.title("Graphique des individus coloré par position")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Graphique des individus coloré par championnat
plt.figure(figsize=(14, 10))

palette2 = plt.get_cmap("Dark2")
compétitions = pca_df["Comp"].unique()
couleurs_comp = dict(zip(compétitions, palette2(range(len(compétitions)))))

for comp in compétitions:
    mask = pca_df["Comp"] == comp
    plt.scatter(pca_df[mask]["Dim1"], pca_df[mask]["Dim2"], c=[couleurs_comp[comp]], label=comp, alpha=0.7)

for i, (x, y, player) in enumerate(zip(pca_df["Dim1"], pca_df["Dim2"], pca_df["Player"])):
    plt.annotate(player, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=6, alpha=0.7)

plt.xlabel("Dimension 1 ({:.1f}%)".format(pca.explained_variance_ratio_[0] * 100))
plt.ylabel("Dimension 2 ({:.1f}%)".format(pca.explained_variance_ratio_[1] * 100))
plt.title("Graphique des individus coloré par championnat")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# AFC

data_crosstab = pd.crosstab(data_quantitative["Age"], data_quantitative["Gls"])

temp = data_crosstab.sub(data_crosstab.mean())
data_scaled = temp.div(data_crosstab.std())
"""
fa = FactorAnalyzer(n_factors = 6, rotation = None)
fa.fit(data_scaled)
ev, v = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1, data_scaled.shape[1] + 1 ), ev)
plt.plot(range(1, data_scaled.shape[1] + 1), ev )
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()
plt.show()
"""