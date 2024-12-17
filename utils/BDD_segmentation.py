import pandas as pd
from pydub import AudioSegment
import os

# Charger les fichiers meta
meta_train = pd.read_csv('meta/meta_train.csv')
meta_test = pd.read_csv('meta/meta_test.csv')

# Fonction pour créer un DataFrame avec les segments
def creer_segments(meta_df):
    segments = []
    for index, row in meta_df.iterrows():
        for i in range(4):
            debut = i * 2+1
            fin = debut + 2
            segments.append({
                'filename': row['audio'],
                'start_time': debut,
                'end_time': fin,
                'label': row['label']
            })
    return pd.DataFrame(segments)

# Créer les DataFrames pour les segments
segments_train = creer_segments(meta_train)
segments_test = creer_segments(meta_test)

# Combiner les deux DataFrames
segments_combines = pd.concat([segments_train, segments_test])



# Sauvegarder dans un seul fichier CSV
fichier_sortie_combines = 'meta/meta_segments_combines_all.csv'
segments_combines.to_csv(fichier_sortie_combines, index=False)

# Compter le nombre de fichiers par label
compte_par_label = segments_combines['label'].value_counts()
print(compte_par_label)


        #DA               DB
        #train test     train dev test
#   rain 1200   300      1200  100  200
#  walck 1200   300      1200  100  200       
#   wind  0      0       2400  200  400 
#    car  0      0       2400  200  400


df_rain = segments_combines[segments_combines['label'] == 'rain']
df_walk = segments_combines[segments_combines['label'] == 'walking']
df_wind = segments_combines[segments_combines['label'] == 'wind']
df_car = segments_combines[segments_combines['label'] == 'car_passing']
bdd_A_train = pd.concat([df_rain[:1200], df_walk[:1200]])
bdd_A_test = pd.concat([df_rain[1200:1500], df_walk[1200:1500]])
bdd_B_train = pd.concat([df_rain[1500:2700], df_walk[1500:2700], df_wind[:2400], df_car[:2400]])
bdd_B_dev = pd.concat([df_rain[2700:2800], df_walk[2700:2800], df_wind[2400:2600], df_car[2400:2600]])
bdd_B_test = pd.concat([df_rain[2800:], df_walk[2800:], df_wind[2600:], df_car[2600:]])

# Sauvegarder les fichiers
bdd_A_train.to_csv('meta/bdd_A_train.csv', index=False)
bdd_A_test.to_csv('meta/bdd_A_test.csv', index=False)
bdd_B_train.to_csv('meta/bdd_B_train.csv', index=False)
bdd_B_dev.to_csv('meta/bdd_B_dev.csv', index=False)
bdd_B_test.to_csv('meta/bdd_B_test.csv', index=False)


bdd_train = pd.concat([df_rain[:2400], df_car[:2400], df_walk[:2400], df_wind[:2400]])
bdd_test = pd.concat([df_rain[2400:], df_car[2400:], df_walk[2400:], df_wind[2400:]])

bdd_train.to_csv('meta/bdd_train.csv', index=False)
bdd_test.to_csv('meta/bdd_test.csv', index=False)