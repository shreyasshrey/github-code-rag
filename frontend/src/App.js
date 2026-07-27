import { useEffect, useRef, useState } from 'react';
import { askQuestion, clearRepositoryApi, ingestRepositoryApi } from './services/api';

import Header from './components/Header';
import RepositoryCard from './components/RepositoryCard';
import ChatMessages from './components/ChatMessages';
import ChatInput from './components/ChatInput';
import './App.css';

function App() {
  const messagesEndRef = useRef(null);

  const [repoUrl, setRepoUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState("");

  const [ingesting, setIngesting] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (!chatLoading && messages.length === 0) {
      return;
    }

    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, chatLoading]);

  async function ingestRepository(e) {
    e.preventDefault();

    if (!repoUrl.trim()) {
      setStatus("Please enter a GitHub repository URL.");
      return;
    }

    setIngesting(true);

    setStatus("Cloning and indexing repository. This may take some time...");

    try {
      const data = await ingestRepositoryApi(repoUrl);

      setStatus(
        `${data.message} Files: ${data.files}. Chunks: ${data.chunks}.`
      );

      setMessages([]);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIngesting(false);
    }
  }

  async function sendQuestion(e) {
    e.preventDefault();

    if (!question.trim()) {
      return;
    }

    const currentQuestion = question;

    setQuestion("");

    setMessages((messages) => [
      ...messages,
      {
        role: "user",
        content: currentQuestion,
      },
    ]);

    setChatLoading(true);

    try {
      const data = await askQuestion(currentQuestion);

      setMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (error) {
      setMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: `Error: ${error.message}`,
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  async function clearRepository() {
    setIngesting(true);

    try {
      const data = await clearRepositoryApi();

      setMessages([]);
      setRepoUrl("");
      setStatus(data.message);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setIngesting(false);
    }
  }

  return (
    <main className="page">
      <section className="container">
        <section className="repo-card">
          <Header />

          <RepositoryCard
            repoUrl={repoUrl}
            setRepoUrl={setRepoUrl}
            ingestRepository={ingestRepository}
            clearRepository={clearRepository}
            ingesting={ingesting}
            chatLoading={chatLoading}
            status={status}
          />
        </section>

        <section className="chat-card">
          <ChatMessages
            messages={messages}
            chatLoading={chatLoading}
            messagesEndRef={messagesEndRef}
          />

          <ChatInput
            question={question}
            setQuestion={setQuestion}
            sendQuestion={sendQuestion}
            chatLoading={chatLoading}
            ingesting={ingesting}
          />
        </section>
      </section>
    </main>
  );
}

export default App;
