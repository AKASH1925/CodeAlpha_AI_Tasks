const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const API_URL = '/api/chat';
function addMessage(text, type = 'bot', metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    const avatarSvg = type === 'bot' 
        ? `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="10" r="1.5" fill="currentColor"/><circle cx="15" cy="10" r="1.5" fill="currentColor"/><path d="M8 14C8 14 9.5 16 12 16C14.5 16 16 14 16 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    let contentHtml = `<p>${escapeHtml(text)}</p>`;
    if (metadata && metadata.confidence > 0) {
        contentHtml += `<span class="confidence-badge">Match: ${metadata.confidence}%</span>`;
    }
    messageDiv.innerHTML = `<div class="message-avatar">${avatarSvg}</div><div class="message-content">${contentHtml}</div>`;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `<div class="message-avatar"><svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><circle cx="9" cy="10" r="1.5" fill="currentColor"/><circle cx="15" cy="10" r="1.5" fill="currentColor"/><path d="M8 14C8 14 9.5 16 12 16C14.5 16 16 14 16 14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></div><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}
function removeTypingIndicator() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}
async function sendMessage(message) {
    addMessage(message, 'user');
    messageInput.value = '';
    sendBtn.disabled = true;
    showTypingIndicator();
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await response.json();
        removeTypingIndicator();
        if (response.ok) {
            addMessage(data.response, 'bot', { confidence: data.confidence, category: data.category });
        } else {
            addMessage(data.error || 'Something went wrong.', 'bot');
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage("Sorry, I couldn't connect to the server.", 'bot');
    } finally {
        sendBtn.disabled = false;
        messageInput.focus();
    }
}
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function clearChat() {
    const messages = chatMessages.querySelectorAll('.message');
    const quickActions = chatMessages.querySelector('.quick-actions');
    messages.forEach((msg, index) => { if (index > 0) msg.remove(); });
    if (quickActions) quickActions.style.display = 'flex';
}
chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const message = messageInput.value.trim();
    if (message) sendMessage(message);
});
clearBtn.addEventListener('click', clearChat);
document.querySelectorAll('.quick-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        sendMessage(btn.dataset.question);
        btn.parentElement.style.display = 'none';
    });
});
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});
messageInput.focus();