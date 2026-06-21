from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot
app = Flask(__name__)
chatbot = FAQChatbot()
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    user_message = data['message'].strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    response = chatbot.get_response(user_message)
    return jsonify({
        'response': response['answer'],
        'confidence': response.get('confidence', 0),
        'matched_question': response.get('question', ''),
        'category': response.get('category', ''),
        'match_found': response.get('match_found', False)
    })
@app.route('/api/faqs', methods=['GET'])
def get_faqs():
    faqs = chatbot.get_all_faqs()
    return jsonify({
        'faqs': faqs,
        'count': len(faqs),
        'categories': chatbot.get_categories()
    })
@app.route('/api/faqs', methods=['POST'])
def add_faq():
    data = request.get_json()
    if not data or 'question' not in data or 'answer' not in data:
        return jsonify({'error': 'Question and answer are required'}), 400
    question = data['question'].strip()
    answer = data['answer'].strip()
    category = data.get('category', 'General').strip()
    chatbot.add_faq(question, answer, category)
    return jsonify({
        'message': 'FAQ added successfully',
        'total_faqs': len(chatbot.get_all_faqs())
    }), 201
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'faqs_loaded': len(chatbot.get_all_faqs()),
        'categories': chatbot.get_categories()
    })
if __name__ == '__main__':
    print(f"Loaded {len(chatbot.get_all_faqs())} FAQs")
    print(f"Categories: {chatbot.get_categories()}")
    app.run(debug=True, host='0.0.0.0', port=5000)