import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.data')

# 1 - Cálculo da média, desvio padrão e variância para o dataset de avaliações completo (por filme);

columns = ['user_id', 'movie_id', 'rating', 'timestamp']

data_frame_movies = pd.read_csv(file_path, sep='\t', names=columns)

print(data_frame_movies['movie_id'].nunique())

movie_statistics = data_frame_movies.groupby(['movie_id'])['rating'].agg([('movie_rating_mean', 'mean'), ('movie_rating_standard_deviation', 'std'), ('movie_rating_variance', 'var')]).reset_index()

print(movie_statistics)

# 2 - Cálculo de média, desvio padrão e variância para cada usuário (armazenar esses valores em novas colunas do dataset);

data_frame_users = pd.read_csv(file_path, sep='\t', names=columns)

user_statistics = data_frame_users.groupby('user_id')['rating'].agg([('user_rating_mean', 'mean'), ('user_rating_standard_deviation', 'std'), ('user_rating_variance', 'var')]).reset_index()

data_frame_users_and_users_statistics = pd.merge(data_frame_users, user_statistics, on='user_id')

print(data_frame_users_and_users_statistics)


# 3 - Encontrar indivíduos que avaliam filmes de forma mais uniforme, i.e., avaliações estão próximo ao valor da média do indivíduo;

uniform_rating_users = user_statistics.sort_values(by='user_rating_standard_deviation')
unique_uniform_rating_users = uniform_rating_users.drop_duplicates(subset='user_id')

print(unique_uniform_rating_users.loc[:, ['user_id', 'user_rating_standard_deviation']].head(10))