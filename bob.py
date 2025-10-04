import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer

data = pd.read_csv("cleaned_football_dataset.csv")

vars = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A']
data_vars = data[vars].dropna()

# Scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data_vars)

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

coeff_labels = list(data_vars.columns)
biplot(pca_res[:, 0:2], np.transpose(pca.components_[0:2, :]), labels=coeff_labels)
plt.show()

# Graphique des individus
pca_df = pd.DataFrame({
    "Dim1": pca_res[:, 0],
    "Dim2": pca_res[:, 1],
    "Player": data.loc[data_vars.index, "Player"],
    "Pos": data.loc[data_vars.index, "Pos"],
    "Comp": data.loc[data_vars.index, "Comp"]
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

# AFC (Analyse par facteurs correspondants)
# Prepare data for AFC: cross-tabulation of Position vs Competition
data['Age_Group'] = pd.cut(data['Age'], 
                         bins=[15, 22, 26, 30, 40], 
                         labels=['Jeune (15-22)', 'Adulte (23-26)', 'Expérimenté (27-30)', 'Vétéran (31-40)'])

# 2. Create goal performance categories
data['Goal_Performance'] = pd.cut(data['Gls'],
                                bins=[0, 5, 10, 15, 35],
                                labels=['Faible (0-5)', 'Moyen (6-10)', 'Bon (11-15)', 'Excellent (16+)'])

data['Assist_Performance'] = pd.cut(data['Ast'],
                                  bins=[-1, 2, 5, 8, 20],
                                  labels=['Faible (0-2)', 'Moyen (3-5)', 'Bon (6-8)', 'Excellent (9+)'])

print("Distribution des catégories créées:")
print("Groupes d'âge:")
print(data['Age_Group'].value_counts().sort_index())
print("\nPerformance de buts:")
print(data['Goal_Performance'].value_counts().sort_index())
print("\nPerformance de passes décisives:")
print(data['Assist_Performance'].value_counts().sort_index())

# Create cross-tabulation for Age Groups vs Goal Performance
cross_tab_age_goals = pd.crosstab(data['Age_Group'], data['Goal_Performance'])

# Perform Correspondence Analysis
ca_age_goals = prince.CA(
    n_components=2,
    n_iter=10,
    random_state=42
)
ca_age_goals = ca_age_goals.fit(cross_tab_age_goals)

# Visualization - Goals vs Age graph
plt.figure(figsize=(14, 10))

# Plot row coordinates (age groups)
row_coords = ca_age_goals.row_coordinates(cross_tab_age_goals)
plt.scatter(row_coords[0], row_coords[1], color='red', s=100, alpha=0.7, label='Groupes d\'âge')

# Plot column coordinates (goal performance)
col_coords = ca_age_goals.column_coordinates(cross_tab_age_goals)
plt.scatter(col_coords[0], col_coords[1], color='blue', s=100, alpha=0.7, label='Performance de buts')

# Add labels for age groups
for age_group, (x, y) in row_coords.iterrows():
    plt.annotate(age_group, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='red', fontweight='bold')

# Add labels for goal performance
for goal_perf, (x, y) in col_coords.iterrows():
    plt.annotate(goal_perf, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='blue', fontweight='bold')

plt.title('AFC: Groupes d\'âge vs Performance de buts', fontsize=14, fontweight='bold')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# AFC 2: Age Groups vs Assist Performance
cross_tab_age_assists = pd.crosstab(data['Age_Group'], data['Assist_Performance'])

# Perform Correspondence Analysis for assists
ca_age_assists = prince.CA(
    n_components=2,
    n_iter=10,
    random_state=42
)
ca_age_assists = ca_age_assists.fit(cross_tab_age_assists)

# Visualization - Assists vs Age graph
plt.figure(figsize=(14, 10))

# Plot row coordinates (age groups)
row_coords_assists = ca_age_assists.row_coordinates(cross_tab_age_assists)
plt.scatter(row_coords_assists[0], row_coords_assists[1], color='red', s=100, alpha=0.7, label='Groupes d\'âge')

# Plot column coordinates (assist performance)
col_coords_assists = ca_age_assists.column_coordinates(cross_tab_age_assists)
plt.scatter(col_coords_assists[0], col_coords_assists[1], color='green', s=100, alpha=0.7, label='Performance de passes')

# Add labels for age groups
for age_group, (x, y) in row_coords_assists.iterrows():
    plt.annotate(age_group, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='red', fontweight='bold')

# Add labels for assist performance
for assist_perf, (x, y) in col_coords_assists.iterrows():
    plt.annotate(assist_perf, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='green', fontweight='bold')

plt.title('AFC: Groupes d\'âge vs Performance de passes décisives', fontsize=14, fontweight='bold')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()