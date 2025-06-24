import numpy as np
from multiprocessing import Pool, cpu_count
import time

def matmul_block(args):
    A_block, B = args
    return np.dot(A_block, B)

if __name__ == "__main__":
    N = 1000
    n_workers = cpu_count()  # Nombre de coeurs dispo
    print(f"Nombre de coeurs disponibles : {n_workers}")

    print(f"Création matrices {N}x{N} en float32...")
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # Découper A en blocs de lignes égaux selon le nombre de workers
    chunk_size = N // n_workers
    A_chunks = [A[i*chunk_size:(i+1)*chunk_size, :] for i in range(n_workers)]

    args = [(chunk, B) for chunk in A_chunks]

    print(f"Calcul du produit matriciel en parallèle sur {n_workers} cœurs...")

    start = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(matmul_block, args)
    end = time.time()

    # Concaténer verticalement les résultats
    C = np.vstack(results)

    print(f"Produit matriciel parallèle terminé en {end - start:.2f} secondes")
    print(f"Taille matrice résultat : {C.shape}")
