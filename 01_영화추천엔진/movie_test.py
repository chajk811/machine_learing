import numpy as np
import pandas as pd
import json


meta = pd.read_csv('the-movies-dataset/movies_metadata.csv', low_memory=False)
# pandas의 read_csv를 통해 데이터를 읽어온다.
# pandas의 df.head()라는 함수를 통해 데이터프레임의 처음 5줄을 출력해 확인하다.
# meta.head()

meta = meta[['id', 'original_title', 'original_language', 'genres']]
meta = meta.rename(columns={'id': 'movieId'})
# df.rename() 칼럼 이름 변경
meta = meta[meta['original_language'] == 'en']
# 평가가 영화로 된것이 많아서, 언어가 영어로 된것만 추린다.
# print(meta.head())

ratings = pd.read_csv('the-movies-dataset/ratings_small.csv')
ratings = ratings[['userId', 'movieId', 'rating']]
# print(ratings.head())

# print(ratings.describe())
# df.describe() : 데이터의 대략적인 개요를 본다.

# 데이터 가공 (str로 되어있음. 연산이 필요하므로 숫자로 변경)
meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')
# pd.to_numeric() : 문자열을 숫자 타입으로 변환한다.

# 장르도 보면 json 형태같지만 str 으로 저장되어 있어 가공해주어야 함. 
def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'','"'))
    
    genres_list = []
    for g in genres:
        genres_list.append(g['name'])
    
    return genres_list

# df.apply() : 각 행에 해당 함수를 적용한다.
meta['genres'] = meta['genres'].apply(parse_genres)
# print(meta.head())

data = pd.merge(ratings, meta, on='movieId', how='inner')
# pd.merge() : 두 개의 데이터프레임을 병합한다. 
# 한 개의 movieId 에 대한 여러 명의 유저들을 각각으로 합친 것?
# print(data.head())

# Pivot Table 생성
# df.pivot_table() : 피벗테이블을 만든다.
matrix = data.pivot_table(index='userId', columns='original_title', values='rating')

# print(matrix.head(20))

# Pearson Correlation
GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc[0]

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])

        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc[0]

            same_count = np.sum(np.isin(input_genres, temp_genres))
            # np.isin() : 배열을 비교하여 똑같은 요소가 있으면 True를 반환한다.
            cor += (GENRE_WEIGHT * same_count)
        
        if np.isnan(cor):
            continue
        else:
            result.append((title,'{:.2f}'.format(cor), temp_genres))

    result.sort(key=lambda r: r[1], reverse=True)

    return result[:n]

recommend_result = recommend('The Dark Knight', matrix, 10, similar_genre=True)

print(pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre']))

# 영화 추천 엔진
# 사용자의 평가를 선형적으로 비교하고
# 비슷한 장르에 가중치를 주는 방식
# 키워드나 같은 감독으로도 가중치의 부여가 가능하다.
# .iloc(0) => .iloc[0] 으로 수정해주었는데 틀린 이유 # https://devpouch.tistory.com/47
# 참고 링크 : https://www.youtube.com/watch?v=mLwMe4KUZz8&t=120s


