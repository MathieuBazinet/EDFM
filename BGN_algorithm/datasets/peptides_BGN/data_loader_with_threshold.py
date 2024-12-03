from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

anion_file_name = "datasets/peptides_BGN/Anionic_16_membranes_VF.csv"
cation_file_name = "datasets/peptides_BGN/Cationic_16_membranes_VF.csv"

DECALAGE = True

decalage = np.array([34, 22])
decalage_full = 56
is_equal = (anion_file_name == cation_file_name)

anion_file_name = "./" + anion_file_name
cation_file_name = "./" + cation_file_name

def load_data(boolean):
    if boolean:
      df = pd.read_csv(cation_file_name, sep=";")
    else:
      df = pd.read_csv(anion_file_name, sep=";")

    if DECALAGE:
      s = df.shape[0]
      df = df.iloc[DECALAGE*(decalage[int(boolean)]*(1-int(is_equal))+ decalage_full*int(is_equal)):, :]


    l = ['R1 A-', 'R2 A-', 'R3 A-', 'R1 C+', 'R2 C+', 'R3 C+']
    y = df[l]
    y.rename(columns={'R1 A-': 'R1_A', 'R2 A-': 'R2_A',
                      'R3 A-': 'R3_A', 'R1 C+': 'R1_C',
                      'R2 C+': 'R2_C', 'R3 C+': 'R3_C'},
             inplace=True)
    l = l + ['membrane', 'Peptides', 'bend_percent', 'turn_percent']
    df = df.drop(columns=l)
    return df, y

def load_data_mean(boolean):
    """
    Retourne les données. On prend la moyenne de la variable réponse.
    :param boolean: True si on veut les données positives, False sinon.
    :return:
    """
    df, y = load_data(boolean)
    if boolean:
        y_pos = y[['R1_C', 'R2_C', 'R2_C']].mean(axis=1)
        return df, y_pos
    else:
        y_neg = y[['R1_A', 'R2_A', 'R3_A']].mean(axis=1)
        return df, y_neg

# def get_peptides_names(boolean):
#       if boolean:
#         df = pd.read_csv(cation_file_name, sep=";")
#       else:
#         df = pd.read_csv(anion_file_name, sep=";")

#       if DECALAGE:
#         df = df.iloc[DECALAGE*(decalage[int(boolean)]*(1-int(is_equal))+ decalage_full*int(is_equal)):, :]
#       return (df['membrane'] + ' + ' + df['Peptides'])

def get_features(X, y, threshold):
  regressor = RandomForestRegressor(random_state=1, n_estimators=1000)
  regressor.fit(X, y)
  feat_importance = pd.DataFrame(regressor.feature_importances_.T, index=X.columns).sort_values(by=0,
                                                                                                        ascending=False)
  new_indexs_pos = feat_importance[feat_importance[0] >= threshold].index
  return new_indexs_pos

def get_selected_data(anion_cation, threshold):
    if anion_cation:
        print(f"Données pour l'expérience avec les cations et le seuil de {threshold}")
    else:
        print(f"Données pour l'expérience avec les anions et le seuil de {threshold}")

    membrane = ['Contact angle', 'hydrophilic pores',
            'Volumetric porosity', 'Zeta-potential', 'Rz', ' Macropores FL']

    peptide = ['mol_weight', 'isoelectric_point',
            'GRAVY', 'm/z_at_pH7.0',
            'Hall Kier Alpha', 'Polar R', 'A', 'D', 'F', 'H',
            'K', 'L', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    features = membrane + peptide

    df, y = load_data(anion_cation)
    if anion_cation:
        y_pos = y[['R1_C', 'R2_C', 'R2_C']].mean(axis=1)
        X, y = df, y_pos
    else:
        y_neg = y[['R1_A', 'R2_A', 'R3_A']].mean(axis=1)
        X, y = df, y_neg

    X_selected = X[features]
    selected_features = get_features(X_selected, y, threshold)
    print("Selected features : " , list(selected_features))
    X_new = X_selected[selected_features]
    return X_new, y

if __name__ == "__main__":
   anion_ou_cation = True #True for cation, false for anion
   data_threshold = 0.02 # 0.02, 0.05 ou 0 (si on veut toutes les features)
   get_selected_data(anion_ou_cation, data_threshold)