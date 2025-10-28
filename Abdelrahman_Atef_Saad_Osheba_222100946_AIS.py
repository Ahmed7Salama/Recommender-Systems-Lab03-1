import numpy as np

def cosine_similarity(user1,user2):

    mask = (user1 !=0) & (user2 !=0)
    if not np.any(mask):
        return 0
    u1 = user1[mask]
    u2 = user2[mask]

    dot_product = 0
    for i in range(len(u1)):
        dot_product += u1[i] * u2[i]

    sum_squares_1 = 0
    for i in range (len(u1)):
        sum_squares_1 += u1[i]**2
    magnitude_1 = np.sqrt(sum_squares_1)

    sum_squares_2 = 0
    for i in range (len(u2)):
        sum_squares_2 += u2[i]**2
    magnitude_2 = np.sqrt(sum_squares_2)

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0
    return dot_product / (magnitude_1 * magnitude_2)

# The Usage Example

User_A = np.array([5, 3, 0, 4, 0, 2])
User_B = np.array([3, 0, 2, 3, 3, 0])
User_C = np.array([4, 3, 4, 3, 5, 3])

Sim_AB = cosine_similarity(User_A, User_B)
Sim_AC = cosine_similarity(User_A, User_C)

print(f"Cosine Similarity Between A and B Score:", Sim_AB)
print(f"Cosine Similarity Between A and C Score:", Sim_AC)
