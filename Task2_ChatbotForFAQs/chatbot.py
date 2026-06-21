import json
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
class FAQChatbot:
    """A chatbot that matches user questions to FAQ answers using NLP."""
    def __init__(self, faqs_path=None):
        if faqs_path is None:
            faqs_path = os.path.join(os.path.dirname(__file__), 'faqs.json')
        self.faqs = self._load_faqs(faqs_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._build_model()
    def _load_faqs(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('faqs', [])
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(processed_tokens)
    def _build_model(self):
        self.processed_questions = [
            self.preprocess_text(faq['question'])
            for faq in self.faqs
        ]
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\S+',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.processed_questions
        )
    def get_response(self, user_question, threshold=0.1):
        if not user_question.strip():
            return {
                'answer': "Please ask a question.",
                'question': '', 'confidence': 0, 'match_found': False
            }
        processed_question = self.preprocess_text(user_question)
        user_tfidf = self.vectorizer.transform([processed_question])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)[0]
        best_match_idx = similarities.argmax()
        best_score = similarities[best_match_idx]
        if best_score >= threshold:
            matched_faq = self.faqs[best_match_idx]
            return {
                'answer': matched_faq['answer'],
                'question': matched_faq['question'],
                'confidence': round(float(best_score) * 100, 2),
                'match_found': True,
                'category': matched_faq.get('category', 'General')
            }
        else:
            return {
                "answer": "I'm sorry, I don't have an answer for that question. "
                          "Please try rephrasing or ask about a different topic.",
                'question': '', 'confidence': 0, 'match_found': False
            }
    def add_faq(self, question, answer, category="General"):
        self.faqs.append({
            'question': question, 'answer': answer, 'category': category
        })
        self._build_model()
    def get_all_faqs(self):
        return self.faqs
    def get_categories(self):
        return list(set(faq.get('category', 'General') for faq in self.faqs))
if __name__ == "__main__":
    chatbot = FAQChatbot()
    print("=" * 60)
    print("FAQ Chatbot - Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to exit, 'faqs' to see all questions")
    print("=" * 60)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'faqs':
            print("\nAvailable FAQs:")
            for i, faq in enumerate(chatbot.get_all_faqs(), 1):
                print(f"  {i}. [{faq.get('category')}] {faq['question']}")
            continue
        elif not user_input:
            continue
        response = chatbot.get_response(user_input)
        print(f"\nBot: {response['answer']}")
        if response['match_found']:
            print(f"     (Confidence: {response['confidence']}%)")