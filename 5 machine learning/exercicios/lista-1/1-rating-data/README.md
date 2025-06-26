
# Análise de Avaliações de Filmes - MovieLens 100k

Este projeto realiza uma análise estatística do dataset `u.data` do repositório [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/), utilizando a biblioteca `pandas` para obter estatísticas por filme e por usuário.

## 📜 Explicação Detalhada do Código `u.data.py`

```python
import pandas as pd
```
Importa a biblioteca `pandas`, usada para manipulação de dados tabulares.

```python
import os
```
Importa a biblioteca `os`, usada para lidar com caminhos de arquivos de forma portátil.

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
```
Obtém o diretório atual onde o script está sendo executado.

```python
file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.data')
```
Cria o caminho para o arquivo `u.data`, assumindo que está em uma pasta acima (`..`) na estrutura de diretórios.

```python
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
```
Define os nomes das colunas do dataset, que não possui cabeçalho.

```python
data_frame_movies = pd.read_csv(file_path, sep='\t', names=columns)
```
Lê o arquivo `u.data` separando as colunas por tabulação (`\t`), aplicando os nomes definidos.

```python
print(data_frame_movies['movie_id'].nunique())
```
Imprime o número de filmes distintos avaliados no dataset.

```python
movie_statistics = data_frame_movies.groupby(['movie_id'])['rating'].agg([
    ('movie_rating_mean', 'mean'),
    ('movie_rating_standard_deviation', 'std'),
    ('movie_rating_variance', 'var')
]).reset_index()
```
Agrupa as avaliações por filme (`movie_id`) e calcula a média, o desvio padrão e a variância das avaliações de cada filme.

```python
print(movie_statistics)
```
Exibe o DataFrame contendo as estatísticas por filme.

```python
data_frame_users = pd.read_csv(file_path, sep='\t', names=columns)
```
Lê novamente o mesmo dataset, agora com foco nos usuários.

```python
user_statistics = data_frame_users.groupby('user_id')['rating'].agg([
    ('user_rating_mean', 'mean'),
    ('user_rating_standard_deviation', 'std'),
    ('user_rating_variance', 'var')
]).reset_index()
```
Agrupa as avaliações por usuário (`user_id`) e calcula as estatísticas para cada um.

```python
data_frame_users_and_users_statistics = pd.merge(data_frame_users, user_statistics, on='user_id')
```
Une o DataFrame original com as estatísticas calculadas, ligando pelo campo `user_id`.

```python
print(data_frame_users_and_users_statistics)
```
Exibe o DataFrame completo com os dados das avaliações e estatísticas de cada usuário.

```python
uniform_rating_users = user_statistics.sort_values(by='user_rating_standard_deviation')
```
Ordena os usuários pelo desvio padrão das suas avaliações — os mais "uniformes" vêm primeiro.

```python
unique_uniform_rating_users = uniform_rating_users.drop_duplicates(subset='user_id')
```
Remove possíveis duplicatas de usuários.

```python
print(unique_uniform_rating_users.loc[:, ['user_id', 'user_rating_standard_deviation']].head(10))
```
Imprime os 10 usuários com menor desvio padrão, ou seja, os mais consistentes nas suas avaliações.

## ▶️ Como Executar o Script

Estrutura de diretórios recomendada:

```
projeto/
├── ml-100k/
│   └── u.item
├── 2-movies-data/
│   └── u.data.py
│   └── README.md
└── requirements.txt
```

Para executar o script:

```
python 1-rating-data/u.data.py
```