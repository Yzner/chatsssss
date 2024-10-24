// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const sendMessage = async () => {
//     if (input.trim() === '') return;
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);
//     try {
//       const response = await axios.post('http://localhost:5000/chat', {
//         user_input: input,
//       });
//       const botMessage = { sender: 'bot', text: response.data.response };
//       setMessages((prevMessages) => [...prevMessages, botMessage]);
//     } catch (error) {
//       const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
//       setMessages((prevMessages) => [...prevMessages, errorMessage]);
//     }
    
//     setInput('');
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter') {
//       sendMessage();
//     }
//   };

//   return (
//     <div className="chat-container">
//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div
//             key={index}
//             className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
//           >
//             {message.text}
//           </div>
//         ))}
//       </div>

//       <div className="input-box">
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyPress={handleKeyPress}
//           placeholder="Type your question..."
//         />
//         <button onClick={sendMessage}>Send</button>
//       </div>
//     </div>
//   );
// };

// export default ChatbotScreen;





import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Ensure to create a separate CSS file for styles

const ChatbotScreen = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (input.trim() === '') return;
    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    try {
      const response = await axios.post('http://192.168.11.188:5000/chat', {
        user_input: input,
      });
      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
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
      <div className="chat-box">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {message.text}
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
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default ChatbotScreen;
