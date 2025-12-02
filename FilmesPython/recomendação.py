"""
Sistema de Recomendação de Filmes com API TMDB
Versão avançada com dados reais e modelo de recomendação baseado e IA
"""

import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class TMDBMovieRecommender:
    def __init__(self, api_key):
        self.api_key = api_key 
        self.base_url = "https://api.themoviedb.org/3"
        self.movies_df = None
        self.similarity_matrix = None
        
    def fetch_popular_movies(self, pages=5):
        """Busca filmes populares da API TMDB"""
        print("🎬 Buscando filmes da API TMDB...")
        movies_list = []
        
        for page in range(1, pages + 1):
            url = f"{self.base_url}/movie/popular"
            params = {
                'api_key': self.api_key,
                'language': 'pt-BR',
                'page': page
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    movies_list.extend(data['results'])
                    print(f"✓ Página {page} carregada")
                else:
                    print(f"✗ Erro na página {page}: {response.status_code}")
            except Exception as e:
                print(f"✗ Erro: {e}")
                
        return movies_list
    
    def get_movie_details(self, movie_id):
        """Busca detalhes completos de um filme"""
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'language': 'pt-BR',
            'append_to_response': 'credits,keywords'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Erro ao buscar detalhes: {e}")
        return None
    
    def process_movies_data(self, movies_list):
        """Processa dados dos filmes para o DataFrame"""
        print("\n🔧 Processando dados dos filmes...")
        
        processed_movies = []
        
        for movie in movies_list:
            # Busca detalhes completos
            details = self.get_movie_details(movie['id'])
            
            if details:
                # Extrai gêneros
                genres = ' '.join([g['name'] for g in details.get('genres', [])])
                
                # Extrai elenco
                cast = []
                if 'credits' in details and 'cast' in details['credits']:
                    cast = [actor['name'] for actor in details['credits']['cast'][:5]]
                
                # Extrai diretor
                director = ''
                if 'credits' in details and 'crew' in details['credits']:
                    for person in details['credits']['crew']:
                        if person['job'] == 'Director':
                            director = person['name']
                            break
                
                # Extrai palavras-chave
                keywords = []
                if 'keywords' in details and 'keywords' in details['keywords']:
                    keywords = [kw['name'] for kw in details['keywords']['keywords'][:10]]
                
                processed_movies.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie.get('overview', ''),
                    'genres': genres,
                    'cast': ' '.join(cast),
                    'director': director,
                    'keywords': ' '.join(keywords),
                    'vote_average': movie.get('vote_average', 0),
                    'popularity': movie.get('popularity', 0),
                    'release_date': movie.get('release_date', ''),
                    'poster_path': movie.get('poster_path', '')
                })
                
        self.movies_df = pd.DataFrame(processed_movies)
        print(f"✓ {len(self.movies_df)} filmes processados!")
        return self.movies_df
    
    def build_recommendation_model(self):
        """Constrói modelo de recomendação avançado"""
        print("\n🧠 Construindo modelo de IA...")
        
        # Combina múltiplas features em uma só
        self.movies_df['combined_features'] = (
            self.movies_df['genres'] + ' ' +
            self.movies_df['keywords'] + ' ' +
            self.movies_df['director'] + ' ' +
            self.movies_df['cast'] + ' ' +
            self.movies_df['overview']
        )
        
        # Vetorização TF-IDF
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Calcula similaridade
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        print("✓ Modelo construído com sucesso!")
        
    def recommend_movies(self, movie_title, n_recommendations=10):
        """Recomenda filmes similares"""
        try:
            # Busca pelo título (case insensitive)
            idx = self.movies_df[
                self.movies_df['title'].str.lower() == movie_title.lower()
            ].index[0]
            
            # Calcula scores de similaridade
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            
            movie_indices = [i[0] for i in sim_scores]
            
            recommendations = self.movies_df.iloc[movie_indices][[
                'title', 'genres', 'director', 'vote_average', 'popularity'
            ]].copy()
            
            recommendations['similarity'] = [round(i[1]*100, 1) for i in sim_scores]
            
            return recommendations
            
        except IndexError:
            return f"❌ Filme '{movie_title}' não encontrado!"
    
    def get_top_rated(self, n=10):
        """Retorna filmes mais bem avaliados"""
        return self.movies_df.nlargest(n, 'vote_average')[[
            'title', 'genres', 'vote_average', 'popularity'
        ]]
    
    def search_by_genre(self, genre, n=10):
        """Busca filmes por gênero"""
        filtered = self.movies_df[
            self.movies_df['genres'].str.contains(genre, case=False, na=False)
        ]
        return filtered.nlargest(n, 'vote_average')[[
            'title', 'genres', 'vote_average'
        ]]

def main():
    print("=" * 70)
    print("🎬 SISTEMA AVANÇADO DE RECOMENDAÇÃO DE FILMES COM IA")
    print("=" * 70)
    api_key = "65ac84ad216a28185485692d2885b5ca"
    
    # Inicializa o sistema
    recommender = TMDBMovieRecommender(api_key)
    
    # Busca filmes populares
    movies = recommender.fetch_popular_movies(pages=3)  # 60 filmes
    
    # Processa dados
    df = recommender.process_movies_data(movies)
    
    # Constrói modelo
    recommender.build_recommendation_model()
    
    # Exemplos de uso
    print("\n" + "=" * 70)
    print("📊 EXEMPLOS DE RECOMENDAÇÕES")
    print("=" * 70)
    
    # Recomendação por filme
    print("\n🎯 Se você gostou de um filme, recomendamos:\n")
    movie_example = df.iloc[0]['title']
    recommendations = recommender.recommend_movies(movie_example, n_recommendations=5)
    print(f"Baseado em: '{movie_example}'\n")
    print(recommendations.to_string(index=False))
    
    # Top filmes
    print("\n" + "=" * 70)
    print("⭐ TOP 10 FILMES MAIS BEM AVALIADOS")
    print("=" * 70)
    top_rated = recommender.get_top_rated(10)
    print(top_rated.to_string(index=False))
    
    # Busca por gênero
    print("\n" + "=" * 70)
    print("🎭 FILMES DE AÇÃO")
    print("=" * 70)
    action_movies = recommender.search_by_genre('Ação', n=5)
    print(action_movies.to_string(index=False))
    
if __name__ == "__main__":
    main()