
# 🎬 Análise de Filmes com Estatísticas e Normalização - MovieLens 100k

Este projeto trabalha com os arquivos `u.item` e `u.data` do dataset MovieLens 100k para extrair informações sobre os filmes, calcular estatísticas e aplicar diferentes técnicas de **normalização** — uma etapa fundamental em projetos de Machine Learning.

---

## 🧠 Explicação do Código `new_dat_frame_movies.py`
### 🔹 1. Importação de bibliotecas

```python
import pandas as pd
import os
```
- `pandas`: permite manipular dados em formato de tabela.
- `os`: permite montar caminhos de arquivos automaticamente.

---

### 🔹 2. Definindo os caminhos dos arquivos

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
uitem_file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.item')
udata_file_path = os.path.join(current_dir, '..', 'ml-100k', 'u.data')
```
- Localiza automaticamente onde está o script e monta os caminhos para os arquivos `u.item` e `u.data`.

---

### 🔹 3. Criando um DataFrame com os gêneros dos filmes

```python
columns = [...]
movies_data_frame = pd.read_csv(...)
genre_columns = movies_data_frame.columns[5:]
```
- Define os nomes das colunas e carrega os dados do arquivo `u.item`.
- Pega apenas as colunas relacionadas aos gêneros dos filmes.

```python
movies_genre = []
for index, row in movies_data_frame.iterrows():
    ...
```
- Cria uma nova coluna chamada `genre`, contendo uma string com os gêneros de cada filme, como `"Action, Comedy"`.

```python
new_movies_data_frame = movies_data_frame.drop(columns=genre_columns)
new_movies_data_frame['genre'] = movies_genre
```
- Remove as colunas de gênero em forma binária e adiciona a versão em texto.

---

### 🔹 4. Calculando estatísticas de avaliação dos filmes

```python
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data_frame_ratings = pd.read_csv(...)
movies_statistics = data_frame_ratings.groupby('movie_id')['rating'].agg(...)
```
- Carrega o arquivo `u.data`, que contém as avaliações dos usuários.
- Agrupa por `movie_id` e calcula:
  - total de avaliações (`count`)
  - soma das notas (`sum`)
  - média (`mean`)
  - maior e menor nota (`max`, `min`)
  - desvio padrão e variância (`std`, `var`)

```python
new_movies_statistics_data_frame = pd.merge(...)
```
- Junta os dados de filmes com os dados de avaliação por `movie_id`.

---

### 🔹 5. Filmes mais e menos avaliados

```python
most_evaluated_movies = ...
least_evaluated_movies = ...
```
- Ordena os filmes pelo número de avaliações e exibe os 10 com mais e os 10 com menos avaliações.

---

### 🔹 6. Normalização dos dados

A normalização é usada para ajustar os valores para uma escala comum, essencial em algoritmos de Machine Learning.

#### ✅ Min-Max

```python
from sklearn.preprocessing import MinMaxScaler
```
- Transforma os valores para um intervalo entre 0 e 1.

#### ✅ Normalização pela média

```python
def mean_normalization(X):
    return (X - X.mean()) / (X.max() - X.min())
```
- Centraliza os dados em torno da média.

#### ✅ Z-Score

```python
from sklearn.preprocessing import StandardScaler
```
- Transforma os dados com média 0 e desvio padrão 1 (padrão estatístico comum).

Em todos os casos, os dados normalizados são mesclados de volta com os dados dos filmes.

---

## ▶️ Como Executar o Script

### Estrutura esperada:

```
projeto/
├── ml-100k/
│   ├── u.item
│   └── u.data
├── 3-new-data-set/
│   └── new_dat_frame_movies.py
│   └── README.md
└── requirements.txt
```

### Passos no terminal

1. Ativar o ambiente virtual:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
```

2. Executar o script:
```bash
python 3-new-data-set/new_dat_frame_movies.py
```

---

## 🧩 Explicações Complementares (Etapas detalhadas)

### 🎯 Construção da coluna "genre" (texto descritivo dos gêneros do filme)

```python
movies_genre = []

for index, row in movies_data_frame.iterrows():
    genre_by_movie = []
    for genre in genre_columns:
        if row[genre] == 1:
            genre_by_movie.append(genre)
    genre_str = ', '.join(genre_by_movie)
    movies_genre.append(genre_str)
```
- Esse trecho percorre **linha por linha** do DataFrame de filmes.
- Para cada filme (`row`), ele verifica quais colunas de gênero possuem valor `1` (indicando que o filme pertence àquele gênero).
- Cria uma **lista de strings** com os nomes dos gêneros e depois transforma em uma única string separada por vírgulas.
- Exemplo: um filme com valores 1 nas colunas `Action` e `Thriller` terá como saída: `"Action, Thriller"`.

```python
new_movies_data_frame = movies_data_frame.drop(columns=genre_columns)
new_movies_data_frame['genre'] = movies_genre
```
- Remove as colunas binárias e substitui por uma única coluna de texto contendo os gêneros.

---

### 📊 Estatísticas por filme: explicação de cada métrica

```python
movies_statistics = data_frame_ratings.groupby('movie_id')['rating'].agg(
    rating_total='count',
    rating_sum='sum',
    rating_mean='mean',
    rating_max_value='max',
    rating_min_value='min',
    rating_standard_deviation='std',
    rating_variance='var'
).reset_index()
```
Para cada filme (`movie_id`), calcula:

- `rating_total`: número de vezes que o filme foi avaliado.
- `rating_sum`: soma total das notas recebidas.
- `rating_mean`: média das notas.
- `rating_max_value`: maior nota recebida.
- `rating_min_value`: menor nota recebida.
- `rating_standard_deviation`: mede o quanto as notas variam da média.
- `rating_variance`: a dispersão dos dados (quadrado do desvio padrão).

Essas métricas ajudam a entender **a popularidade, consistência e variabilidade** das avaliações de cada filme.

---

### 🔃 Normalizações com mais detalhes

#### ✅ Min-Max

```python
scaler = MinMaxScaler()
normalized_data_min_max = scaler.fit_transform(...)
```
- Transforma os valores para uma escala de **0 a 1** com base no valor mínimo e máximo de cada coluna.
- Útil quando você quer manter a proporção dos dados, mas padronizar a escala.

#### ✅ Normalização pela média

```python
def mean_normalization(X):
    return (X - X.mean()) / (X.max() - X.min())
```
- Centraliza os dados em torno de zero considerando a **média e o intervalo (máx - mín)**.
- Isso ajuda a evitar que os valores com maiores magnitudes dominem os algoritmos de aprendizado.

#### ✅ Z-score

```python
StandardScaler()
```
- Calcula: `z = (x - média) / desvio padrão`
- Resultado: dados com **média 0 e desvio padrão 1**.
- Útil para algoritmos que assumem distribuição normal (como regressão logística, SVM, etc).

---

### 🧩 Junção final dos dados normalizados

Em todas as normalizações (MinMax, Média, Z-score), o resultado é um novo DataFrame que:

- Contém os valores normalizados (`rating_total`, `rating_sum` em nova escala)
- É unido de volta ao conjunto original usando `pd.merge(..., on='movie_id')` para preservar a identidade de cada filme.

Isso garante que cada filme tenha todas as suas métricas e normalizações combinadas em um só lugar.

---

