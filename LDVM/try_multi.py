import time
import multiprocessing

# Fonction simulant une tâche lourde
def carre(x):
    time.sleep(0.11)  # Simule une tâche CPU ou I/O
    return x * x

# Exécution séquentielle
def execution_sequentielle(donnees):
    resultats = []
    debut = time.time()
    for x in donnees:
        resultats.append(carre(x))
    fin = time.time()
    print("Séquentielle:", resultats)
    print("Durée séquentielle:", round(fin - debut, 2), "secondes")

# Exécution parallèle avec multiprocessing
def execution_parallele(donnees):
    debut = time.time()
    with multiprocessing.Pool(processes=len(donnees)) as pool:
        resultats = pool.map(carre, donnees)
    fin = time.time()
    print("Parallèle :", resultats)
    print("Durée parallèle :", round(fin - debut, 2), "secondes")

# Point d'entrée du script
if __name__ == "__main__":
    donnees = [1, 2, 3, 4, 5]

    print("==== Exécution séquentielle ====")
    execution_sequentielle(donnees)

    print("\n==== Exécution parallèle ====")
    execution_parallele(donnees)
