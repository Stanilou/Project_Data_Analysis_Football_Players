import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import prince
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer

################################################################
###                    CLEAN DES DONNÉES                     ###
################################################################

df = pd.read_csv('joueurs_football_dataset.csv')

# Garde uniquement le nom de la ligue
df['Comp'] = df['Comp'].str.replace(r'^[a-z]{2,3}\s+', '', regex=True)

# Garde une position simple ou double unique
def standardize_positions(pos):
    positions = sorted(pos.split(','))
    return ','.join(positions)

df['Pos'] = df['Pos'].apply(standardize_positions)

# Save
df.to_csv('cleaned_football_dataset.csv', index=False)

################################################################
###                 CHARGEMENT DES DONNÉES                   ###
################################################################

data = pd.read_csv("cleaned_football_dataset.csv")

vars = ['Age', 'MP', 'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A']
data_vars = data[vars].dropna()

# Scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data_vars)


################################################################
###               ANALYSE QUANTITATIVE : ACP                 ###
################################################################


################################################################
###               TABLEAU DES VALEURS PROPRES                ###
################################################################

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


################################################################
###                   CERCLE DE CORRÉLATION                  ###
################################################################

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


################################################################
###                 GRAPHIQUE DES INDIVIDUS                  ###
################################################################

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


################################################################
###       GRAPHIQUE DES INDIVIDUS COLORÉ PAR POSITION        ###
################################################################

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


################################################################
###     GRAPHIQUE DES INDIVIDUS COLORÉ PAR CHAMPIONNAT       ###
################################################################

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


################################################################
###                 ANALYSE CATÉGORIELLE : AFC               ###
################################################################


################################################################
###        GRAPHIQUE DES ASSOCIATIONS PAR ÂGES ET BUTS       ###
################################################################

data['Age_Group'] = pd.cut(data['Age'], 
                         bins=[15, 22, 26, 30, 40], 
                         labels=['Jeune (15-22)', 'Adulte (23-26)', 'Expérimenté (27-30)', 'Vétéran (31-40)'])

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

cross_tab_age_goals = pd.crosstab(data['Age_Group'], data['Goal_Performance'])

ca_age_goals = prince.CA(
    n_components=2,
    n_iter=10,
    random_state=42
)
ca_age_goals = ca_age_goals.fit(cross_tab_age_goals)

plt.figure(figsize=(14, 10))

row_coords = ca_age_goals.row_coordinates(cross_tab_age_goals)
plt.scatter(row_coords[0], row_coords[1], color='red', s=100, alpha=0.7, label='Groupes d\'âge')

col_coords = ca_age_goals.column_coordinates(cross_tab_age_goals)
plt.scatter(col_coords[0], col_coords[1], color='blue', s=100, alpha=0.7, label='Performance de buts')

for age_group, (x, y) in row_coords.iterrows():
    plt.annotate(age_group, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='red', fontweight='bold')

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


################################################################
###      GRAPHIQUE DES ASSOCIATIONS PAR ÂGES ET PASSES       ###
################################################################

cross_tab_age_assists = pd.crosstab(data['Age_Group'], data['Assist_Performance'])

ca_age_assists = prince.CA(
    n_components=2,
    n_iter=10,
    random_state=42
)
ca_age_assists = ca_age_assists.fit(cross_tab_age_assists)

plt.figure(figsize=(14, 10))

row_coords_assists = ca_age_assists.row_coordinates(cross_tab_age_assists)
plt.scatter(row_coords_assists[0], row_coords_assists[1], color='red', s=100, alpha=0.7, label='Groupes d\'âge')

col_coords_assists = ca_age_assists.column_coordinates(cross_tab_age_assists)
plt.scatter(col_coords_assists[0], col_coords_assists[1], color='green', s=100, alpha=0.7, label='Performance de passes')

for age_group, (x, y) in row_coords_assists.iterrows():
    plt.annotate(age_group, (x, y), xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='red', fontweight='bold')

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


################################################################
###                    ANALYSE MIXTE : ACM                   ###
################################################################

variables_qualitatives = ['Pos', 'Comp', 'Age_Group', 'Goal_Performance', 'Assist_Performance']
data_acm = data[variables_qualitatives].dropna()

encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(data_acm)
X = pd.DataFrame(X, columns=encoder.get_feature_names_out(variables_qualitatives))

nb_dimensions = 2 
svd = TruncatedSVD(n_components=nb_dimensions)
components = svd.fit_transform(X)

variance_expliquee = svd.explained_variance_ratio_

valeurs_propres = pd.DataFrame({"Dimension": ["Dim" + str(i+1) for i in range(nb_dimensions)], "% variance expliquée": np.round(variance_expliquee * 100, 2)})
print("\nTableau des variances expliquées (ACM) :")
print(valeurs_propres) 


################################################################
###                GRAPHIQUE DES INDIVIDUS (ACM)             ###
################################################################

groupes_ages = data.loc[data_acm.index, 'Age_Group']

tranche_ages = groupes_ages.unique() 
couleurs = cm.tab10(np.linspace(0, 1, len(tranche_ages)))  
couleur_age = {age: couleurs[i] for i, age in enumerate(tranche_ages)}

plt.figure(figsize=(10,8))

for age in tranche_ages:
    idx = groupes_ages[groupes_ages == age].index 
    plt.scatter(
        components[[data_acm.index.get_loc(i) for i in idx], 0],
        components[[data_acm.index.get_loc(i) for i in idx], 1],
        alpha=0.7,
        s=50,
        label=age,
        color=couleur_age[age]
    )

plt.title("ACM – Représentation des individus par tranche d'âge")
plt.xlabel(f"Dimension 1 ({variance_expliquee[0]*100:.1f}%)")
plt.ylabel(f"Dimension 2 ({variance_expliquee[1]*100:.1f}%)")
plt.grid(alpha=0.3)
plt.legend(loc='upper left')
plt.show()