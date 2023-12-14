import numpy as np
import pandas as pd
from IPython.display import display
import math

# Read data from txt files
movies = open('HW8/hw8_movies.txt').read().splitlines()
ids = open('HW8/hw8_ids.txt').read().splitlines()
ratings = np.loadtxt('HW8/hw8_ratings.txt', dtype='str')
probR = np.loadtxt('HW8/hw8_probR_init.txt')
probZ = np.loadtxt('HW8/hw8_probZ_init.txt')

# part a
class_ratings = np.zeros((258, 76))
for i in range(ratings.shape[0]):
    for j in range(ratings.shape[1]):
        if ratings[i][j] == '?':
            class_ratings[i][j] -= 1.0
        else:
            class_ratings[i][j] += float(ratings[i][j])

def calculate_popularity(rating):
    df = pd.DataFrame({'Movie': movies, 'popularity rating': rating})
    df = df.sort_values(by='popularity rating')
    df = df.reset_index(drop=True)
    return df

popularity = np.sum(class_ratings == 1, axis=0) / np.sum(class_ratings >= 0, axis=0)
popularity_sorted = np.argsort(popularity)
display(calculate_popularity(popularity))

# part e
mask1 = (class_ratings == 1) + 0
mask0 = (class_ratings == 0) + 0
mask_1 = (class_ratings == -1) + 0

def calculate_product_log(pr, pz):
    return np.exp(mask1.dot(np.log(pr)) + mask0.dot(np.log(1 - pr)))


def calculate_product(pr):
    result = []
    for i in range(class_ratings.shape[0]):
        temp1 = mask1[i][:, np.newaxis]
        temp0 = mask0[i][:, np.newaxis]
        temp = pr * temp1 + (1 - pr) * temp0
        product = np.prod(temp, axis=0, where=(temp > 0))
        result.append(product)
    result = np.array(result)
    return result

def calculate_posterior(pr, pz):
    product = calculate_product(pr)
    denominator = (product.dot(pz))[:, np.newaxis]
    numerator = product * pz
    return numerator / denominator

def calculate_loss(pr, pz):
    product = calculate_product(pr)
    return 1 / class_ratings.shape[0] * np.sum(np.log(product.dot(pz)))

def update_parameters(pr, pz):
    posterior = calculate_posterior(pr, pz)
    pz_update = 1 / class_ratings.shape[0] * np.sum(posterior, axis=0)
    denominator = pz_update * class_ratings.shape[0]
    numerator = mask1.T.dot(posterior) + mask_1.T.dot(posterior) * pr
    pr_update = numerator / denominator
    return pr_update, pz_update

initial_loss = calculate_loss(probR, probZ)
log_list = [initial_loss]
print('iteration 0\tlog-likelihood  %.4f' % initial_loss)

for i in range(1, 257):
    probR_new, probZ_new = update_parameters(probR, probZ)
    log_likelihood = calculate_loss(probR_new, probZ_new)
    log_list.append(log_likelihood)
    probR= probR_new
    probZ = probZ_new
    if math.log(i, 2).is_integer():
        print('iteration %d\tlog-likelihood  %.4f' % (i, log_likelihood))

# part f
user_index = ids.index('A15443798')
posterior_user = calculate_posterior(probR, probZ)[user_index][:, np.newaxis]
expected_rating_user = (probR * mask_1[user_index][:, np.newaxis]).dot(posterior_user)
df_user = pd.DataFrame({'Movie': movies, 'my probability': expected_rating_user.flatten()})
df_user = df_user.sort_values(by='my probability')
index_names_user = df_user[df_user['my probability'] == 0].index
df_user.drop(index_names_user, inplace=True)
df_user = df_user.reset_index(drop=True)
print(df_user)