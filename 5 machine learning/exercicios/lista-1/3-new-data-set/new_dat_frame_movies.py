import pandas as pd
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
uitem_file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.item')
udata_file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.data')

################################################################################################################################################################################

# 1 - Criar DataFrame que contenha informações sobre o gênero do filme:

columns = [
    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

movies_data_frame = pd.read_csv(uitem_file_path, sep='|', names=columns, encoding='latin-1', engine='python')
genre_columns = movies_data_frame.columns[5:]

movies_genre = []

for index, row in movies_data_frame.iterrows():
    genre_by_movie = [] 
    for genre in genre_columns:
        if row[genre] == 1:
            genre_by_movie.append(genre)
    genre_str = ', '.join(genre_by_movie)

    movies_genre.append(genre_str)

new_movies_data_frame = movies_data_frame.drop(columns=genre_columns)
new_movies_data_frame['genre'] = movies_genre

#print(new_movies_data_frame)

################################################################################################################################################################################

# 2 - Adicionar colunas que armazenem dados para o total de avaliações, a soma das avaliações, média, valor máximo (e mínimo), desvio padrão e variância;

columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data_frame_ratings = pd.read_csv(udata_file_path, sep='\t', names=columns)
movies_statistics = data_frame_ratings.groupby('movie_id')['rating'].agg(
    rating_total='count',
    rating_sum='sum',
    rating_mean='mean',
    rating_max_value='max',
    rating_min_value='min',
    rating_standard_deviation='std',
    rating_variance='var'
).reset_index()

new_movies_statistics_data_frame =  pd.merge(new_movies_data_frame, movies_statistics, on='movie_id', how='left')

#print(new_movies_statistics_data_frame.iloc[:, 4:])

################################################################################################################################################################################

# 3 -Mostrar filmes com maior (e menor) número de avaliações;
# Maior número

most_evaluated_movies = new_movies_statistics_data_frame.sort_values(by='rating_total', ascending=False).head(10)#
print(most_evaluated_movies[['movie_id','rating_total']])

# Menor número
least_evaluated_movies = new_movies_statistics_data_frame.sort_values(by='rating_total', ascending=True).head(10)
print(least_evaluated_movies[['movie_id','rating_total']])

################################################################################################################################################################################

# 4 - Normalização é uma das tarefas mais importantes quando estamos preparando um dataset para utilizar algoritmos de Machine Learning. Implementar as seguintes estratégias de normalização:
# Normalização min-max

from sklearn.preprocessing import MinMaxScaler  # normalization

columns_for_normalization = ['rating_total', 'rating_sum']
scaler = MinMaxScaler()

normalized_data_min_max = pd.DataFrame(
    scaler.fit_transform(new_movies_statistics_data_frame[columns_for_normalization]),
        columns=[f'{col}_minmax' for col in columns_for_normalization]
)

normalized_data_min_max['movie_id'] = new_movies_statistics_data_frame['movie_id']
print(normalized_data_min_max['movie_id'])

new_movies_data_frame_dropped_columns = new_movies_statistics_data_frame.drop(columns=columns_for_normalization)  

new_movies_statistics_data_frame_min_max = pd.merge(new_movies_data_frame_dropped_columns, normalized_data_min_max, on='movie_id' )

print(new_movies_statistics_data_frame_min_max.iloc[:, 4:])

# Normalização pela média

from sklearn.preprocessing import FunctionTransformer

def mean_normalization(X):
    return (X - X.mean()) / (X.max() - X.min())

mean_normalization_transformer = FunctionTransformer(mean_normalization)
normalized_data_mean = mean_normalization_transformer.fit_transform(new_movies_statistics_data_frame[columns_for_normalization])

normalized_data_mean['movie_id'] =  new_movies_statistics_data_frame['movie_id']

new_movies_statistics_data_frame_mean = pd.merge(new_movies_data_frame_dropped_columns, normalized_data_mean, on='movie_id' )

print(new_movies_statistics_data_frame_mean.iloc[:, 4:])


# Normalização Z-score 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_data_z_score_array = scaler.fit_transform(new_movies_statistics_data_frame[columns_for_normalization])

normalized_data_z_score = pd.DataFrame(
    normalized_data_z_score_array,
    columns=[f'{col}_zscore' for col in columns_for_normalization]
)

normalized_data_z_score['movie_id'] = new_movies_statistics_data_frame['movie_id']

new_movies_statistics_data_frame_z_score = pd.merge(new_movies_data_frame_dropped_columns, normalized_data_z_score, on='movie_id' )

print(new_movies_statistics_data_frame_z_score.iloc[:, 4:])
