import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configuration de l'affichage
plt.style.use('default')
sns.set_palette("husl")

# Chargement des données
df = pd.read_csv('cleaned_dataset.csv')

# 1. DISTRIBUTION DES EFFETS DE LA MUSIQUE
plt.figure(figsize=(10, 6))
effects_dist = df['Music effects'].value_counts()
colors = ['#2ecc71', '#95a5a6', '#e74c3c']  # Vert pour Improve, Rouge pour Worsen, Gris pour No effect

plt.subplot(2, 2, 1)
plt.pie(effects_dist.values, labels=effects_dist.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Distribution des Effets de la Musique\nsur la Santé Mentale', fontsize=12, fontweight='bold')

# 2. EFFETS PAR GENRE MUSICAL FAVORI
plt.subplot(2, 2, 2)
genre_effects = pd.crosstab(df['Fav genre'], df['Music effects'], normalize='index') * 100
genre_effects = genre_effects.sort_values('Improve', ascending=False).head(10)

genre_effects.plot(kind='bar', stacked=True, ax=plt.gca(), 
                  color=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title('Effets par Genre Musical Favori (Top 10)', fontsize=12, fontweight='bold')
plt.xlabel('Genre Musical')
plt.ylabel('Pourcentage (%)')
plt.legend(title='Effet', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')

# 3. CORRÉLATION ENTRE HEURES D'ÉCOUTE ET EFFETS
plt.subplot(2, 2, 3)
effects_mapping = {'Improve': 2, 'No effect': 1, 'Worsen': 0}
df['effects_numeric'] = df['Music effects'].map(effects_mapping)

# Boxplot des heures d'écoute par effet
sns.boxplot(data=df, x='Music effects', y='Hours per day', 
            order=['Worsen', 'No effect', 'Improve'],
            palette=['#e74c3c', '#95a5a6', '#2ecc71'])
plt.title('Heures d\'Écoute Quotidiennes\npar Effet sur la Santé Mentale', 
          fontsize=12, fontweight='bold')
plt.xlabel('Effet de la Musique')
plt.ylabel('Heures par Jour')

# 4. CORRÉLATION AVEC LES TROUBLES MENTAUX
plt.subplot(2, 2, 4)
mental_health_vars = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

# Calcul des scores moyens par effet
mental_health_means = df.groupby('Music effects')[mental_health_vars].mean()

mental_health_means.plot(kind='bar', ax=plt.gca(), 
                        color=['#3498db', '#9b59b6', '#e67e22', '#f1c40f'])
plt.title('Scores Moyens des Troubles Mentaux\npar Effet de la Musique', 
          fontsize=12, fontweight='bold')
plt.xlabel('Effet de la Musique')
plt.ylabel('Score Moyen (0-10)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# 5. ANALYSE PAR ÂGE ET EFFETS
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Distribution d'âge par effet
sns.violinplot(data=df, x='Music effects', y='Age', 
               order=['Worsen', 'No effect', 'Improve'],
               palette=['#e74c3c', '#95a5a6', '#2ecc71'])
plt.title('Distribution d\'Âge par Effet de la Musique', fontsize=12, fontweight='bold')
plt.xlabel('Effet de la Musique')
plt.ylabel('Âge')

plt.subplot(1, 2, 2)
# Effets par groupe d'âge
df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 50, 100], 
                        labels=['<18', '18-25', '26-35', '36-50', '50+'])
age_effects = pd.crosstab(df['age_group'], df['Music effects'], normalize='index') * 100

age_effects.plot(kind='bar', stacked=True, ax=plt.gca(),
                color=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title('Effets de la Musique par Groupe d\'Âge', fontsize=12, fontweight='bold')
plt.xlabel('Groupe d\'Âge')
plt.ylabel('Pourcentage (%)')
plt.legend(title='Effet', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 6. ANALYSE DES MUSICIENS VS NON-MUSICIENS
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Effets pour les instrumentistes
instrumentalist_effects = pd.crosstab(df['Instrumentalist'], df['Music effects'], normalize='index') * 100
instrumentalist_effects.plot(kind='bar', stacked=True, ax=plt.gca(),
                           color=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title('Effets de la Musique pour les Instrumentistes', fontsize=12, fontweight='bold')
plt.xlabel('Est Instrumentiste')
plt.ylabel('Pourcentage (%)')
plt.legend(title='Effet', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks([0, 1], ['Non', 'Oui'], rotation=0)

plt.subplot(1, 2, 2)
# Effets pour les compositeurs
composer_effects = pd.crosstab(df['Composer'], df['Music effects'], normalize='index') * 100
composer_effects.plot(kind='bar', stacked=True, ax=plt.gca(),
                    color=['#2ecc71', '#95a5a6', '#e74c3c'])
plt.title('Effets de la Musique pour les Compositeurs', fontsize=12, fontweight='bold')
plt.xlabel('Est Compositeur')
plt.ylabel('Pourcentage (%)')
plt.legend(title='Effet', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks([0, 1], ['Non', 'Oui'], rotation=0)

plt.tight_layout()
plt.show()

# CALCUL DES KPIs
print("="*50)
print("INDICATEURS CLÉS (KPIs)")
print("="*50)

# KPI 1: Taux d'amélioration global
improvement_rate = (df['Music effects'] == 'Improve').mean() * 100
print(f"1. Taux d'amélioration global: {improvement_rate:.1f}%")

# KPI 2: Genres les plus bénéfiques
genre_improvement = df[df['Music effects'] == 'Improve']['Fav genre'].value_counts().head(3)
print(f"\n2. Genres les plus bénéfiques:")
for genre, count in genre_improvement.items():
    rate = (df[df['Fav genre'] == genre]['Music effects'] == 'Improve').mean() * 100
    print(f"   - {genre}: {rate:.1f}% d'amélioration")

# KPI 3: Impact de la pratique musicale
instrumentalist_improve = df[df['Instrumentalist'] == 'Yes']['Music effects'].value_counts(normalize=True)['Improve'] * 100
non_instrumentalist_improve = df[df['Instrumentalist'] == 'No']['Music effects'].value_counts(normalize=True)['Improve'] * 100
print(f"\n3. Impact de la pratique instrumentale:")
print(f"   - Instrumentistes: {instrumentalist_improve:.1f}% d'amélioration")
print(f"   - Non-instrumentistes: {non_instrumentalist_improve:.1f}% d'amélioration")

# KPI 4: Corrélation avec le temps d'écoute
hours_correlation = df['Hours per day'].corr(df['effects_numeric'])
print(f"\n4. Corrélation temps d'écoute/effet: {hours_correlation:.3f}")

# KPI 5: Groupes d'âge les plus sensibles
age_group_improve = df.groupby('age_group')['effects_numeric'].mean().sort_values(ascending=False)
print(f"\n5. Groupes d'âge les plus sensibles (score moyen):")
for age_group, score in age_group_improve.head(3).items():
    print(f"   - {age_group}: {score:.2f}")

# TEST STATISTIQUE : Chi2 pour l'association genre/effet
print(f"\n6. Test statistique (Chi2):")
contingency_table = pd.crosstab(df['Fav genre'], df['Music effects'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"   - Test d'indépendance genre/effet: p-value = {p_value:.4f}")
if p_value < 0.05:
    print("     → Association significative entre le genre et l'effet")
else:
    print("     → Pas d'association significative")

print("\n" + "="*50)