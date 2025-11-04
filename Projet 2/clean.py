import pandas as pd

def clean_data(file_path):
    # Load
    df = pd.read_csv(file_path)

    # Suppression des doublons
    df.drop_duplicates(inplace=True)

    # Suppression des lignes avec une valeur manquante
    df.dropna(inplace=True)

    # Suppression de la colone 'Timestamp'
    if ('Timestamp' in df.columns):
        df.drop(columns = ['Timestamp'], inplace = True)

    # Suppression de la colonne 'Primary streaming service'
    if ('Primary streaming service' in df.columns):
        df.drop(columns = ['Primary streaming service'], inplace = True)

    # Suppression de la colonne 'Permissions'
    if('Permissions' in df.columns):
        df.drop(columns = ['Permissions'], inplace = True)

    # Ajout de la colonne 'Id'
    df.insert(0, 'Id', range(1, len(df) + 1))

    # Save
    df.to_csv('cleaned_dataset.csv', index=False)

clean_data('mxmh_survey_results.csv')