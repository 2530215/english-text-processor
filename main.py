import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

# nltk 데이터 다운로드 (최초 실행 시 필요)
nltk.download('punkt')
nltk.download('stopwords')

def analyze_text(text):
    """영어 지문을 분석하고 결과를 반환합니다."""

    # 1. 사용자 입력 (이미 텍스트로 주어졌다고 가정)
    print("입력 텍스트:\n", text)

    # 2. 단어 단위 처리/문장 단위 처리
    sentence_list = sent_tokenize(text)
    word_list = [word.lower() for sentence in sentence_list for word in word_tokenize(sentence) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    word_list = [word for word in word_list if word not in stop_words]

    print("\n단어 리스트 (일부):\n", word_list[:10])
    print("\n문장 리스트 (일부):\n", sentence_list[:2])

    # 3. Word2Vec 처리 (pre-trained model 이용)
    # pre-trained 모델 로드 (예: GoogleNews-vectors-negative300.bin.gz)
    # 실제 모델 경로에 맞춰 수정해야 합니다.
    try:
        model = Word2Vec.load("word2vec_model.bin") # 사용자 지정 모델 로드 예시
    except FileNotFoundError:
        print("Word2Vec 모델 파일이 없습니다. pre-trained 모델을 다운로드하거나 직접 학습시켜야 합니다.")
        return None

    word_vector_list = [model.wv[word] for word in word_list if word in model.wv]
    sentence_vector_list = [np.mean([model.wv[word] for word in word_tokenize(sentence.lower()) if word in model.wv and word not in stop_words] or [np.zeros(model.vector_size)], axis=0) for sentence in sentence_list]
    text_vector = np.mean(sentence_vector_list, axis=0) if sentence_vector_list else np.zeros(model.vector_size)

    print("\n단어 벡터값 리스트 (일부):\n", word_vector_list[:1])
    print("\n문장 벡터값 리스트 (일부):\n", sentence_vector_list[:1])
    print("\n글 전체 벡터값:\n", text_vector[:10])

    # 4. 핵심 키워드/문장 뽑기
    # - 가장 많이 등장하는 단어
    word_counts = Counter(word_list)
    frequent_word_list = word_counts.most_common(10) # 상위 10개

    # - word2vec 처리 결과 글 전체 맥락과 관련 높은 단어 + 문장
    def most_similar_n(vector, topn=5):
        """주어진 벡터와 가장 유사한 단어들을 반환합니다."""
        if not isinstance(vector, np.ndarray) or vector.ndim != 1 or vector.shape[0] != model.vector_size:
            return []
        return model.wv.most_similar(positive=[vector], topn=topn)

    important_word_list = most_similar_n(text_vector, topn=10)

    def sentence_similarity(sent_vec, text_vec):
        """문장 벡터와 글 전체 벡터의 코사인 유사도를 계산합니다."""
        if np.linalg.norm(sent_vec) == 0 or np.linalg.norm(text_vec) == 0:
            return 0
        return np.dot(sent_vec, text_vec) / (np.linalg.norm(sent_vec) * np.linalg.norm(text_vec))

    sentence_similarity_scores = [(sentence, sentence_similarity(vec, text_vector)) for sentence, vec in zip(sentence_list, sentence_vector_list)]
    important_sentence_list = sorted(sentence_similarity_scores, key=lambda item: item[1], reverse=True)[:3] # 유사도 상위 3개 문장

    print("\n가장 많이 등장하는 단어 리스트:\n", frequent_word_list)
    print("\n맥락상 중요한 단어 리스트:\n", important_word_list)
    print("\n맥락상 중요한 문장 리스트:\n", important_sentence_list)

    # 5. 문장별 벡터값 비교 통해 문단을 총 3개로 나누기
    if len(sentence_vector_list) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10) # n_init 경고 방지
        cluster_labels = kmeans.fit_predict(sentence_vector_list)
        paragraph_sentence = [[] for _ in range(3)]
        for i, label in enumerate(cluster_labels):
            paragraph_sentence[label].append(sentence_list[i])
    else:
        paragraph_sentence = [sentence_list] # 문장 수가 3개 미만이면 그대로 반환

    print("\n문단별로 나눠진 글:\n", paragraph_sentence)

    return {
        "word_list": word_list,
        "sentence_list": sentence_list,
        "word_vector_list": word_vector_list,
        "sentence_vector_list": sentence_vector_list,
        "frequent_word_list": frequent_word_list,
        "important_word_list": important_word_list,
        "important_sentence_list": important_sentence_list,
        "paragraph_sentence": paragraph_sentence
    }

if __name__ == "__main__":
    sample_text = input("영어 지문을 입력해 주세요")
    results = analyze_text(sample_text)
    # 필요하다면 결과 활용
    # print(results["frequent_word_list"])
