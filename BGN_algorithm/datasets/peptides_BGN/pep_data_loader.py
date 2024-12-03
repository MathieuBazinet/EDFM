import numpy as np
import pandas as pd

anion_file_name = "datasets/peptides_BGN/Anionic_16 membranes_VF.csv"
cation_file_name = "datasets/peptides_BGN/Cationic_16 membranes_VF.csv"
is_equal = (anion_file_name == cation_file_name)

DECALAGE = True

decalage = np.array([34, 22])
decalage_full = 56

def load_data(boolean):
    if boolean:
        df = pd.read_csv(cation_file_name, sep=";")
    else:
        df = pd.read_csv(anion_file_name, sep=";")

    if DECALAGE:
        df = df.iloc[DECALAGE*(decalage[int(boolean)]*(1-int(is_equal))+ decalage_full*int(is_equal)):, :]


    l = ['R1 A-', 'R2 A-', 'R3 A-', 'R1 C+', 'R2 C+', 'R3 C+']
    y = df[l]
    y.rename(columns={'R1 A-': 'R1_A', 'R2 A-': 'R2_A',
                      'R3 A-': 'R3_A', 'R1 C+': 'R1_C',
                      'R2 C+': 'R2_C', 'R3 C+': 'R3_C'},
             inplace=True)
    l = l + ['membrane', 'Peptides', 'bend_percent', 'turn_percent']
    df = df.drop(columns=l)

    membrane = ['Contact angle', 'hydrophilic pores',
                'Volumetric porosity', 'Zeta-potential', 'Rz', ' Macropores FL']

    peptide = ['mol_weight', 'isoelectric_point',
               'GRAVY', 'm/z_at_pH7.0',
               'Hall Kier Alpha', 'Polar R', 'A', 'D', 'F', 'H',
               'K', 'L', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    features = membrane + peptide
    df = df[features]

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
