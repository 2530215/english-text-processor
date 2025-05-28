import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import gensim.downloader as api  # Word2Vec 모델 다운로드를 위해 추가

# nltk 데이터 다운로드 (최초 실행 시 필요)
try:
    nltk.data.find('corpora/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Word2Vec 모델 로드 (전역 변수로 관리)
@st.cache_resource
def load_word2vec_model():
    try:
        model = Word2Vec.load("word2vec_model.bin")
        return model
    except FileNotFoundError:
        st.warning("Word2Vec 모델 파일(word2vec_model.bin)을 찾을 수 없습니다. GoogleNews 모델을 다운로드합니다.")
        try:
            model = api.load('word2vec-google-news-300')
            model.save("word2vec_model.bin")  # 로컬에 저장 (다음 실행부터 사용)
            return model
        except Exception as e:
            st.error(f"Word2Vec 모델 로드 실패: {e}")
            return None

word2vec_model = load_word2vec_model()

def analyze_text(text, model):
    """영어 지문을 분석하고 결과를 딕셔너리로 반환합니다."""
    if model is None:
        return None

    # 2. 단어 단위 처리/문장 단위 처리
    sentence_list = sent_tokenize(text)
    word_list = [word.lower() for sentence in sentence_list for word in word_tokenize(sentence) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    word_list = [word for word in word_list if word not in stop_words]

    # 3. Word2Vec 처리
    word_vector_list = [model.wv[word] for word in word_list if word in model.wv]
    sentence_vector_list = [np.mean([model.wv[word] for word in word_tokenize(sentence.lower()) if word in model.wv and word not in stop_words] or [np.zeros(model.vector_size)], axis=0) for sentence in sentence_list]
    text_vector = np.mean(sentence_vector_list, axis=0) if sentence_vector_list else np.zeros(model.vector_size)

    # 4. 핵심 키워드/문장 뽑기
    word_counts = Counter(word_list)
    frequent_word_list = word_counts.most_common(10)

    def most_similar_n(vector, topn=5):
        if not isinstance(vector, np.ndarray) or vector.ndim != 1 or vector.shape[0] != model.vector_size:
            return []
        return model.wv.most_similar(positive=[vector], topn=topn)

    important_word_list = most_similar_n(text_vector, topn=10)

    def sentence_similarity(sent_vec, text_vec):
        if np.linalg.norm(sent_vec) == 0 or np.linalg.norm(text_vec) == 0:
            return 0
        return np.dot(sent_vec, text_vec) / (np.linalg.norm(sent_vec) * np.linalg.norm(text_vec))

    sentence_similarity_scores = [(sentence, sentence_similarity(vec, text_vector)) for sentence, vec in zip(sentence_list, sentence_vector_list)]
    important_sentence_list = sorted(sentence_similarity_scores, key=lambda item: item[1], reverse=True)[:3]

    # 5. 문장별 벡터값 비교 통해 문단을 총 3개로 나누기
    paragraph_sentence = []
    if len(sentence_vector_list) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(sentence_vector_list)
        paragraph_sentence = [[] for _ in range(3)]
        for i, label in enumerate(cluster_labels):
            paragraph_sentence[label].append(sentence_list[i])
    else:
        paragraph_sentence = [sentence_list]

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

st.title("영어 지문 분석기")

english_text = st.text_area("분석할 영어 지문을 입력하세요:", height=200)

if st.button("분석 시작"):
    if english_text:
        with st.spinner("텍스트 분석 중..."):
            analysis_results = analyze_text(english_text, word2vec_model)

        if analysis_results:
            st.subheader("분석 결과")

            st.write("### 단어 리스트 (일부)")
            st.write(analysis_results["word_list"][:20])

            st.write("### 문장 리스트 (일부)")
            st.write(analysis_results["sentence_list"][:5])

            st.write("### 가장 많이 등장하는 단어")
            frequent_words = ", ".join([f"{word} ({count})" for word, count in analysis_results["frequent_word_list"]])
            st.write(frequent_words)

            st.write("### 맥락상 중요한 단어")
            important_words = ", ".join([f"{word} ({similarity:.2f})" for word, similarity in analysis_results["important_word_list"]])
            st.write(important_words)

            st.write("### 맥락상 중요한 문장")
            for sentence, similarity in analysis_results["important_sentence_list"]:
                st.write(f"- {sentence} (유사도: {similarity:.2f})")

            st.write("### 문단 분할 결과")
            if analysis_results["paragraph_sentence"]:
                for i, paragraph in enumerate(analysis_results["paragraph_sentence"]):
                    st.write(f"**문단 {i+1}**")
                    for sentence in paragraph:
                        st.write(sentence)
            else:
                st.write("문장 수가 부족하여 문단을 나눌 수 없습니다.")
        else:
            st.error("텍스트 분석에 실패했습니다. Word2Vec 모델 로드 여부를 확인해주세요.")
    else:
        st.warning("분석할 영어 지문을 입력해주세요.")
