// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// const App = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const sendMessage = async () => {
//     if (input.trim() === '') return;

//     // Add the user's message to the chat history
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);

//     // Send user input to the backend
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

//     // Clear input field after sending
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

// export default App;




import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ChatbotScreen from './ChatbotScreen';
import AdminScreen from './AdminScreen';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/chatbotscreen" />} />
        <Route path="/chatbotscreen" element={<ChatbotScreen />} />
        <Route path="/admin" element={<AdminScreen />} />
        <Route path="*" element={<h2>404 Page Not Found</h2>} />
      </Routes>
    </Router>
  );
};

export default App;
