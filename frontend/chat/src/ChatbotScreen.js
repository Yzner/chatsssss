
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './styles/App.css';
import botIcon from "./website/Pictures/logocs.png";
import { IoSend } from 'react-icons/io5';



const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

const ChatbotScreen = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('');

  useEffect(() => {
    const savedSessionId = localStorage.getItem('sessionId');
    const userId = localStorage.getItem('userId');

    if (!savedSessionId) {
      const newSessionId = generateSessionId();
      localStorage.setItem('sessionId', newSessionId);
      setSessionId(newSessionId);
    } else {
      setSessionId(savedSessionId);
    }

    const fetchConversationHistory = async () => {
      try {
        const response = await axios.get('http://localhost:5000/conversation_history', {
          params: { session_id: savedSessionId || sessionId, user_id: userId },
        });
        
        console.log("Fetched conversation history:", response.data);

        if (Array.isArray(response.data)) {
          setMessages(response.data.map((msg) => ({
            sender: msg.sender,
            text: msg.message,
          })));
        }
      } catch (error) {
        console.error("Error fetching conversation history:", error);
      }
    };

    fetchConversationHistory();
  }, [sessionId]);

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const userId = localStorage.getItem('userId');
      const response = await axios.post('http://localhost:5000/chat', {
        user_input: input,
        session_id: sessionId,
        user_id: userId,
      });

      console.log("Bot response:", response.data);
      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    setInput('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <header className="chatbot-header">
        <div className="header-content">
          <img src={botIcon} alt="Bot-header" className="bot-header" />
          <h1 className="header-text">Ask.CS</h1>
        </div>
      </header>
      <div className="chat-box">
        {messages.map((message, index) => (
          <div key={index} className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}>
            {message.sender === 'bot' && (
              <img src={botIcon} alt="Bot Icon" className="bot-icon" />
            )}
            <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
              {message.text}
            </div>
          </div>
        ))}
      </div>

      <div className="input-box">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your question..."
        />
        <button onClick={sendMessage} className="send-button">
         <IoSend size={24} color="green" />
        </button>

      </div>
    </div>
  );
};

export default ChatbotScreen;
