import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Baixando os recursos do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Leitura do arquivo CSV
df = pd.read_csv('C:\\Users\\Dados_SPAM.csv')

# Pré-processamento do texto
def preprocess_text(text):
    # Tokenização
    words = word_tokenize(text)
    
    # Remoção de stopwords e pontuações
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    return words

# Aplicando o pré-processamento à coluna 'Message'
df['Processed_Message'] = df['Message'].apply(preprocess_text)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['Processed_Message'], df['Category'], test_size=0.2, random_state=42)

# Treinamento do modelo Word2Vec
word2vec_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)

# Função para obter a média dos vetores de palavras para representar cada documento
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Obtendo vetores de documentos para treino
X_train_vectors = np.array([get_doc_vector(tokens, word2vec_model) for tokens in X_train])
X_test_vectors = np.array([get_doc_vector(tokens, word2vec_model) for tokens in X_test])

# Treinamento do modelo de classificação (usando Random Forest neste exemplo)
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_vectors, y_train)

# Avaliação do modelo
accuracy = classifier.score(X_test_vectors, y_test)
print(f'Acurácia do modelo: {accuracy}')

# Exemplo de classificação para uma nova mensagem
new_message = "Dear Student, Due to the holiday, next Friday (03/11), there will be no classes."     # Adicionar aqui a nova mensagem
new_message_tokens = preprocess_text(new_message)
new_message_vector = get_doc_vector(new_message_tokens, word2vec_model)
predicted_category = classifier.predict([new_message_vector])[0]

print(f"A categoria prevista para a nova mensagem é: {predicted_category}")

