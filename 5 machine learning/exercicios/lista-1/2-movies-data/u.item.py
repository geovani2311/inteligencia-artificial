import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.item')

columns = [
    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


# 1 - Criar DataFrame que contenha informações sobre filmes:
movies_data_frame = pd.read_csv(file_path, sep='|', names=columns, encoding='latin-1', engine='python')

# 2 - Identificar qual gênero de filme possui o maior número de exemplos;
number_of_movies_by_gender = movies_data_frame.iloc[:, 5:].sum()
gender_with_more_movies = number_of_movies_by_gender.idxmax()

print(f"The gender with more movies is {gender_with_more_movies}")

# 3 - Verificar se existem dados faltando

amount_of_missing_data = movies_data_frame.isnull().sum().sum()

print(amount_of_missing_data)