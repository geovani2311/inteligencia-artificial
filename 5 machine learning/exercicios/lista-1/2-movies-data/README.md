
# Análise de Gêneros de Filmes - MovieLens 100k

Este projeto realiza uma análise estatística do arquivo `u.item` do repositório [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/), extraindo informações sobre os gêneros e dados dos filmes.

## 📜 Explicação Detalhada do Código `u.item.py`

```python
import pandas as pd
```
Importa a biblioteca `pandas` para manipulação de dados.

```python
import os
```
Importa a biblioteca `os` para manipular caminhos de arquivos de forma portátil.

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
```
Determina o diretório onde o script atual está localizado.

```python
file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.item')
```
Constrói o caminho para o arquivo `u.item`, assumindo que ele está na pasta `ml-100k`, um nível acima.

```python
columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
```
Define manualmente os nomes das colunas, já que o arquivo não possui cabeçalho.

```python
movies_data_frame = pd.read_csv(file_path, sep='|', names=columns, encoding='latin-1', engine='python')
```
Lê o arquivo `u.item` utilizando `|` como separador, com codificação `latin-1` e usando o engine `python` (necessário por causa da codificação).

```python
number_of_movies_by_gender = movies_data_frame.iloc[:, 5:].sum()
```
Soma as colunas correspondentes aos gêneros de filmes (a partir da 6ª coluna em diante) para contar quantos filmes existem por gênero.

```python
gender_with_more_movies = number_of_movies_by_gender.idxmax()
```
Identifica o gênero com a maior quantidade de filmes.

```python
print(f"The gender with more movies is {gender_with_more_movies}")
```
Exibe no console qual gênero tem mais filmes.

```python
amount_of_missing_data = movies_data_frame.isnull().sum().sum()
```
Calcula a quantidade total de dados ausentes no DataFrame.

```python
print(amount_of_missing_data)
```
Exibe a quantidade de dados faltantes (se houver).

## ▶️ Como Executar o Script

Estrutura de diretórios recomendada:

```
projeto/
├── ml-100k/
│   └── u.item
├── 2-movies-data/
│   └── u.item.py
│   └── README.md
└── requirements.txt
```

Para executar o script:

```
python 2-movies-data/u.item.py
```