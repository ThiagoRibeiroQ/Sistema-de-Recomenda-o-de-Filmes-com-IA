"""
Sistema de Recomendação com Deep Learning
Usando Redes Neurais para recomendações avançadas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class DeepLearningRecommender:
    def __init__(self):
        self.model = None
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        
    def create_sample_data(self, n_users=100, n_movies=50, n_ratings=2000):
        """dataset para treinamento"""
        print("📊 Criando dataset de treinamento...")
        
        np.random.seed(42)
        
        # Gera avaliações aleatórias
        user_ids = np.random.randint(0, n_users, n_ratings)
        movie_ids = np.random.randint(0, n_movies, n_ratings)
        
        # Simula preferências dos usuários
        ratings = []
        for user, movie in zip(user_ids, movie_ids):
            base_rating = 3.0
            user_bias = (user % 3) * 0.5
            movie_bias = (movie % 4) * 0.3
            noise = np.random.normal(0, 0.5)
            
            rating = base_rating + user_bias + movie_bias + noise
            rating = np.clip(rating, 1, 5)
            ratings.append(rating)
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'movie_id': movie_ids,
            'rating': ratings
        })
        
        # Remove duplicatas
        df = df.drop_duplicates(subset=['user_id', 'movie_id'])
        
        print(f"✓ Dataset criado: {len(df)} avaliações")
        print(f"  - {df['user_id'].nunique()} usuários")
        print(f"  - {df['movie_id'].nunique()} filmes")
        
        return df
    
    def build_neural_network(self, n_users, n_movies, embedding_size=50):
        """Constrói rede neural para recomendação"""
        print("\n🧠 Construindo Rede Neural...")
        
        user_input = layers.Input(shape=(1,), name='user_input')
        movie_input = layers.Input(shape=(1,), name='movie_input')
        
        #aprendizado de representações
        user_embedding = layers.Embedding(
            n_users, 
            embedding_size,
            name='user_embedding'
        )(user_input)
        movie_embedding = layers.Embedding(
            n_movies, 
            embedding_size,
            name='movie_embedding'
        )(movie_input)
        
        # Flatten
        user_vec = layers.Flatten()(user_embedding)
        movie_vec = layers.Flatten()(movie_embedding)
        
        # Concatena embeddings
        concat = layers.Concatenate()([user_vec, movie_vec])
        
        # Camadas densas (fully connected)
        dense1 = layers.Dense(128, activation='relu')(concat)
        dropout1 = layers.Dropout(0.3)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.2)(dense2)
        
        dense3 = layers.Dense(32, activation='relu')(dropout2)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='rating')(dense3)
        
        # Compila modelo
        model = keras.Model(
            inputs=[user_input, movie_input],
            outputs=output
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("✓ Arquitetura da Rede Neural:")
        model.summary()
        
        return model
    
    def train_model(self, df, epochs=20, batch_size=64):
        """Treina o modelo de deep learning"""
        print("\n🎓 Treinando modelo...")
        
        # Codifica IDs
        df['user_encoded'] = self.user_encoder.fit_transform(df['user_id'])
        df['movie_encoded'] = self.movie_encoder.fit_transform(df['movie_id'])
        
        # Normaliza ratings
        df['rating_scaled'] = self.scaler.fit_transform(df[['rating']])
        
        # Split treino/teste
        X_user = df['user_encoded'].values
        X_movie = df['movie_encoded'].values
        y = df['rating_scaled'].values
        
        X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = \
            train_test_split(X_user, X_movie, y, test_size=0.2, random_state=42)
        
        # Cria modelo
        n_users = df['user_encoded'].nunique()
        n_movies = df['movie_encoded'].nunique()
        
        self.model = self.build_neural_network(n_users, n_movies)
        
        # Callback para early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Treina
        history = self.model.fit(
            [X_user_train, X_movie_train],
            y_train,
            validation_data=([X_user_test, X_movie_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Avalia
        test_loss, test_mae = self.model.evaluate(
            [X_user_test, X_movie_test],
            y_test,
            verbose=0
        )
        
        print(f"\n✓ Treinamento concluído!")
        print(f"  MAE no teste: {test_mae:.4f}")
        print(f"  (Erro médio: ~{test_mae * (5-1):.2f} estrelas)")
        
        return history
    
    def predict_rating(self, user_id, movie_id):
        """Prevê rating para um usuário e filme"""
        try:
            user_enc = self.user_encoder.transform([user_id])[0]
            movie_enc = self.movie_encoder.transform([movie_id])[0]
            
            prediction_scaled = self.model.predict(
                [[user_enc], [movie_enc]],
                verbose=0
            )[0][0]
            
            # Desnormaliza
            prediction = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
            prediction = np.clip(prediction, 1, 5)
            
            return round(prediction, 2)
        except:
            return None
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """Recomenda filmes para um usuário"""
        try:
            user_enc = self.user_encoder.transform([user_id])[0]
            
            # Testa todos os filmes
            all_movies = list(range(len(self.movie_encoder.classes_)))
            user_array = np.full(len(all_movies), user_enc)
            
            predictions_scaled = self.model.predict(
                [user_array, all_movies],
                verbose=0
            ).flatten()
            
            # Desnormaliza
            predictions = self.scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            
            # Ordena por rating previsto
            top_indices = np.argsort(predictions)[::-1][:n_recommendations]
            top_movies = self.movie_encoder.inverse_transform(top_indices)
            top_ratings = predictions[top_indices]
            
            recommendations = pd.DataFrame({
                'movie_id': top_movies,
                'predicted_rating': np.round(top_ratings, 2)
            })
            
            return recommendations
        except:
            return None
    
    def get_similar_users(self, user_id, n_similar=5):
        """Encontra usuários similares baseado nos embeddings"""
        try:
            # Extrai embedding layer
            embedding_model = keras.Model(
                inputs=self.model.input[0],
                outputs=self.model.get_layer('user_embedding').output
            )
            
            user_enc = self.user_encoder.transform([user_id])[0]
            user_embedding = embedding_model.predict([[user_enc]], verbose=0)[0]
            
            # Calcula similaridade com todos usuários
            all_users = list(range(len(self.user_encoder.classes_)))
            all_embeddings = embedding_model.predict(
                np.array(all_users).reshape(-1, 1),
                verbose=0
            )
            
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                user_embedding.reshape(1, -1),
                all_embeddings.reshape(len(all_users), -1)
            )[0]
            
            # Top N similares (excluindo o próprio usuário)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            similar_users = self.user_encoder.inverse_transform(similar_indices)
            
            return list(zip(similar_users, similarities[similar_indices]))
        except:
            return None

def main():
    print("=" * 70)
    print("🧠 SISTEMA DE RECOMENDAÇÃO COM DEEP LEARNING")
    print("=" * 70)
    
    # Inicializa
    recommender = DeepLearningRecommender()
    
    # Cria e treina
    df = recommender.create_sample_data(n_users=100, n_movies=50, n_ratings=2000)
    history = recommender.train_model(df, epochs=15, batch_size=64)
    
    # Testes
    print("\n" + "=" * 70)
    print("🎯 EXEMPLOS DE PREDIÇÕES")
    print("=" * 70)
    
    test_user = 5
    test_movie = 10
    
    print(f"\n1. Predição de rating:")
    predicted = recommender.predict_rating(test_user, test_movie)
    print(f"   Usuário {test_user} + Filme {test_movie} = {predicted}⭐")
    
    print(f"\n2. Top 5 recomendações para Usuário {test_user}:")
    recommendations = recommender.recommend_for_user(test_user, n_recommendations=5)
    if recommendations is not None:
        print(recommendations.to_string(index=False))
    else:
        print("Nenhuma recomendação encontrada")
    
    print(f"\n3. Usuários similares ao Usuário {test_user}:")
    similar = recommender.get_similar_users(test_user, n_similar=3)
    if similar:
        for user, similarity in similar:
            print(f"   Usuário {user}: {similarity:.3f} de similaridade")
    
    print("\n" + "=" * 70)
    print("💡 PRÓXIMOS PASSOS:")
    print("=" * 70)
    print("""
    1. Integrar com dados reais da API TMDB
    2. Adicionar mais features (gêneros, diretores, etc)
    3. Usar técnicas de regularização avançadas
    4. Implementar Autoencoders ou VAE
    5. Testar arquiteturas como NCF (Neural Collaborative Filtering)
    6. Salvar e carregar modelos treinados
    """)
    
    # Salva modelo
    print("\n💾 Salvando modelo...")
    recommender.model.save('movie_recommender_model.h5')
    print("✓ Modelo salvo como 'movie_recommender_model.h5'")

if __name__ == "__main__":
    main()