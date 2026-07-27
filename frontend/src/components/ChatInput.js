import React from 'react'

const ChatInput = ({
  question,
  setQuestion,
  sendQuestion,
  chatLoading,
  ingesting,
}) => {
  return (
    <form onSubmit={sendQuestion} className="chat-form">
      <input
        placeholder="Ask a question about the indexed code..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <button disabled={chatLoading || ingesting}>
        {chatLoading ? "Thinking..." : "Ask"}
      </button>
    </form>
  );
}

export default ChatInput