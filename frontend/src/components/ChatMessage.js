import React from 'react'
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const ChatMessage = ({ message }) => {
  const { role, content, sources } = message;

  return (
    <div className={`message ${role}`}>
      <div className="role">
        {role === "user" ? "You" : "Assistant"}
      </div>

      <div className="content">
        {role === "assistant" ? (
          <ReactMarkdown
            components={{
              code({
                node,
                inline,
                className,
                children,
                ...props
              }) {
                const match = /language-(\w+)/.exec(
                  className || ""
                );

                return !inline && match ? (
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {content}
          </ReactMarkdown>
        ) : (
          content
        )}
      </div>

      {sources?.length > 0 && (
        <div className="sources">
          Sources:

          {sources.map((source) => (
            <span key={source}>{source}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export default ChatMessage
