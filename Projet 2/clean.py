import pandas as pd

def clean_data(file_path):
    # Load
    df = pd.read_csv(file_path)

    # Supprime les doublons
    df.drop_duplicates(inplace=True)

    # Supprimme les lignes avec une valeur manquante
    df.dropna(inplace=True)

    # Save
    df.to_csv('cleaned_dataset.csv', index=False)

clean_data('mxmh_survey_results.csv')