import React from 'react'
import ChatMessage from './ChatMessage';
import LoadingMessage from './LoadingMessage';

const ChatMessages = ({
  messages,
  chatLoading,
  messagesEndRef,
}) => {
  return (
    <div className="card messages">
      {messages.length === 0 && (
        <div className="empty">
          Index a repository and ask a question such as:
          <br />

          <strong>
            Where is authentication logic implemented?
          </strong>
        </div>
      )}

      {messages.map((message, index) => (
        <ChatMessage
          key={index}
          message={message}
        />
      ))}

      {chatLoading && <LoadingMessage />}

      <div ref={messagesEndRef} />
    </div>
  );
}

export default ChatMessages