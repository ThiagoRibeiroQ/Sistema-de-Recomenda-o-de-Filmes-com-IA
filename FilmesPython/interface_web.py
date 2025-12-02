"""
Interface Web para Sistema de Recomendação de Filmes
Usando Streamlit para criar uma aplicação interativa
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(
    page_title="Sistema de Recomendação de Filmes",
    page_icon="🎬",
    layout="wide"
)

class MovieRecommenderApp:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        
    @st.cache_data
    def fetch_movies(_self, pages=5):
        """Busca filmes com cache"""
        movies = []
        for page in range(1, pages + 1):
            url = f"{_self.base_url}/movie/popular"
            params = {
                'api_key': _self.api_key,
                'language': 'pt-BR',
                'page': page
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies.extend(response.json()['results'])
        return movies
    
    @st.cache_data
    def get_movie_details(_self, movie_id):
        """Busca detalhes com cache"""
        url = f"{_self.base_url}/movie/{movie_id}"
        params = {
            'api_key': _self.api_key,
            'language': 'pt-BR',
            'append_to_response': 'credits,keywords'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None
    
    @st.cache_data
    def process_data(_self, movies):
        """Processa dados com cache"""
        processed = []
        progress_bar = st.progress(0)
        
        for i, movie in enumerate(movies):
            details = _self.get_movie_details(movie['id'])
            if details:
                genres = ' '.join([g['name'] for g in details.get('genres', [])])
                cast = []
                if 'credits' in details:
                    cast = [a['name'] for a in details['credits'].get('cast', [])[:5]]
                
                director = ''
                if 'credits' in details:
                    for person in details['credits'].get('crew', []):
                        if person['job'] == 'Director':
                            director = person['name']
                            break
                
                processed.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie.get('overview', ''),
                    'genres': genres,
                    'cast': ' '.join(cast),
                    'director': director,
                    'vote_average': movie.get('vote_average', 0),
                    'popularity': movie.get('popularity', 0),
                    'poster_path': movie.get('poster_path', ''),
                    'release_date': movie.get('release_date', '')
                })
            progress_bar.progress((i + 1) / len(movies))
        
        return pd.DataFrame(processed)
    
    @st.cache_data
    def build_model(_self, df):
        """Constrói modelo com cache"""
        df['features'] = (
            df['genres'] + ' ' +
            df['director'] + ' ' +
            df['cast'] + ' ' +
            df['overview']
        )
        
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['features'])
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return similarity
    
    def recommend(self, df, similarity_matrix, movie_title, n=6):
        """Gera recomendações"""
        try:
            idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]
            indices = [i[0] for i in sim_scores]
            return df.iloc[indices]
        except:
            return None

def main():
    # Header
    st.title("🎬 Sistema de Recomendação de Filmes com IA")
    st.markdown("---")
    
    # Sidebar para configuração
    st.sidebar.header("⚙️ Configurações")
    api_key = st.sidebar.text_input(
        "API Key TMDB",
        type="password",
        help="Obtenha em: https://www.themoviedb.org/settings/api",
        key="api_key_input_unique"
    )
    
    if not api_key:
        st.warning("⚠️ Insira sua API Key do TMDB na barra lateral para começar!")
        st.info("""
        ### 📝 Como obter sua API Key (grátis):
        1. Acesse [TMDB](https://www.themoviedb.org/signup)
        2. Crie uma conta gratuita
        3. Vá em Configurações > API
        4. Solicite uma chave API (Developer)
        5. Cole aqui na barra lateral
        """)
        return
    
    # Inicializa app
    app = MovieRecommenderApp(api_key)
    
    # Carrega dados
    with st.spinner("🔄 Carregando filmes..."):
        movies = app.fetch_movies(pages=3)
        df = app.process_data(movies)
        similarity_matrix = app.build_model(df)
    
    st.success(f"✅ {len(df)} filmes carregados!")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Recomendações", "⭐ Top Filmes", "🔍 Buscar"])
    
    # Tab 1: Recomendações
    with tab1:
        st.header("Recomendações Personalizadas")
        
        selected_movie = st.selectbox(
            "Selecione um filme que você gostou:",
            df['title'].values,
            key="movie_selector_unique"
        )
        
        n_recs = st.slider("Quantas recomendações?", 3, 10, 6, key="num_recs_slider")
        
        if st.button("🚀 Recomendar", type="primary", key="recommend_button"):
            recommendations = app.recommend(df, similarity_matrix, selected_movie, n_recs)
            
            if recommendations is not None:
                st.subheader(f"Filmes similares a '{selected_movie}':")
                
                cols = st.columns(3)
                for i, (_, movie) in enumerate(recommendations.iterrows()):
                    with cols[i % 3]:
                        if movie['poster_path']:
                            st.image(
                                f"{app.image_base_url}{movie['poster_path']}",
                                use_container_width=True
                            )
                        st.markdown(f"**{movie['title']}**")
                        st.write(f"⭐ {movie['vote_average']:.1f}/10")
                        st.write(f"🎭 {movie['genres']}")
                        with st.expander("Ver mais"):
                            st.write(f"📅 {movie['release_date']}")
                            st.write(f"🎬 {movie['director']}")
                            st.write(movie['overview'][:200] + "...")
            else:
                st.error("Filme não encontrado!")
    
    # Tab 2: Top Filmes
    with tab2:
        st.header("Top Filmes Mais Bem Avaliados")
        
        top_n = st.slider("Quantos filmes mostrar?", 5, 20, 10, key="top_n_slider")
        top_movies = df.nlargest(top_n, 'vote_average')
        
        for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if movie['poster_path']:
                    st.image(
                        f"{app.image_base_url}{movie['poster_path']}",
                        use_container_width=True
                    )
            
            with col2:
                st.subheader(f"{i}. {movie['title']}")
                st.write(f"⭐ **Avaliação:** {movie['vote_average']:.1f}/10")
                st.write(f"🎭 **Gêneros:** {movie['genres']}")
                st.write(f"🎬 **Diretor:** {movie['director']}")
                st.write(f"📅 **Lançamento:** {movie['release_date']}")
                with st.expander("Sinopse"):
                    st.write(movie['overview'])
            
            st.markdown("---")
    
    # Buscar
    with tab3:
        st.header("Buscar Filmes")
        
        search_term = st.text_input("🔍 Digite o nome do filme:", key="search_input_unique")
        
        if search_term:
            results = df[df['title'].str.contains(search_term, case=False, na=False)]
            
            st.write(f"Encontrados {len(results)} resultados:")
            
            for _, movie in results.iterrows():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if movie['poster_path']:
                        st.image(
                            f"{app.image_base_url}{movie['poster_path']}",
                            use_container_width=True
                        )
                
                with col2:
                    st.subheader(movie['title'])
                    st.write(f"{movie['vote_average']:.1f}/10")
                    st.write(f" {movie['genres']}")
                    st.write(movie['overview'][:150] + "...")
                
                st.markdown("---")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Sobre o Sistema:**
    
    Este sistema usa Machine Learning para recomendar filmes baseado em:
    - Gêneros
    - Elenco
    - Diretor
    - Palavras-chave
    - Sinopse
    
    Algoritmo: TF-IDF + Cosine Similarity
    """)

if __name__ == "__main__":
    main()