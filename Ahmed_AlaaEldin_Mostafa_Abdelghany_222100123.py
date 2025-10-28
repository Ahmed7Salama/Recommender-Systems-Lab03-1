import numpy as np


def cosine_similarity(user1, user2):
    mask = (user1 != 0) & (user2 != 0)
    if not np.any(mask):
        return 0

    u1 = user1[mask]
    u2 = user2[mask]

    dot_product = np.sum(u1 * u2)
    magnitude1 = np.sqrt(np.sum(u1 ** 2))
    magnitude2 = np.sqrt(np.sum(u2 ** 2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# Examples
user_A = np.array([5, 3, 0, 4, 0, 2])
user_B = np.array([3, 0, 2, 3, 3, 0])
user_C = np.array([4, 3, 4, 3, 5, 3])

similarity_AB = cosine_similarity(user_A, user_B)
similarity_AC = cosine_similarity(user_A, user_C)

print("Similarity (A, B):", similarity_AB)
print("Similarity (A, C):", similarity_AC)
