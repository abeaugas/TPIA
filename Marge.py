import numpy as np
from scipy import stats

# Exemple de 5 échantillons
samples = [0.97,0.9717,0.9683,0.9667,0.9483]

# Calculer la moyenne et l'écart-type des échantillons
mean = np.mean(samples)
std_dev = np.std(samples, ddof=1)  # ddof=1 pour un échantillon

# Choisir le niveau de confiance (par exemple, 95%)
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)

# Calculer la marge d'erreur
n = len(samples)
margin_of_error = z_score * (std_dev / np.sqrt(n))

print(f"Moyenne: {mean}")
print(f"Écart-type: {std_dev}")
print(f"Marge d'erreur: {margin_of_error}")