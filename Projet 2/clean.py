import pandas as pd

file_path = 'dataset.csv'

def clean_data(file_path):

    # Chargement du fichier CSV
    df = pd.read_csv(file_path)

    # Suppression des doublons et des lignes avec des valeurs manquantes
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Suppression des colones 'Timestamp', 'Primary streaming service' et 'Permissions'
    if ('Timestamp' in df.columns):
        df.drop(columns = ['Timestamp'], inplace = True)

    if ('Primary streaming service' in df.columns):
        df.drop(columns = ['Primary streaming service'], inplace = True)

    if('Permissions' in df.columns):
        df.drop(columns = ['Permissions'], inplace = True)

    # Ajout de la colonne 'Id'
    df.insert(0, 'Id', range(1, len(df) + 1))

    # Enregistrement du DataFrame nettoyé dans un nouveau fichier CSV
    df.to_csv('cleaned_dataset.csv', index=False)

clean_data(file_path)