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





// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css'; // Ensure to create a separate CSS file for styles

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const sendMessage = async () => {
//     if (input.trim() === '') return;
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);
//     try {
//       const response = await axios.post('http://192.168.11.188:5000/chat', {
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






// import React, { useState } from 'react';
// import axios from 'axios';
// import './styles/App.css';
// import botIcon from './chat.png';
// // import {FontAwesomeIcon} from '@fortawesome/react-fontawesome';
// // import {faBars} from '@fortawesome/free-solid-svg-icons';

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const sendMessage = async () => {
//     if (input.trim() === '') return;
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);
//     try {
//       const response = await axios.post('http://192.168.11.188:5000/chat', {
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
//       <header className="chatbot-header"> 
//         <div className="header-content">
//           <img src={botIcon} alt="Bot-header" className="bot-header" />
//           <h1 className="header-text">Chatbot</h1>
//         </div>
//         {/* <button className='nav-button'>
//           <FontAwesomeIcon icon ={faBars}/>
//         </button> */}
//       </header>
//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div
//             key={index}
//             className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}
//           >
//             {message.sender === 'bot' && (
//               <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             )}
//             <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
//               {message.text}
//             </div>
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








// // chatbotscreen.js

// import React, { useState } from 'react';
// import axios from 'axios';
// import './styles/App.css';
// import botIcon from './chat.png';

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');

//   const sendMessage = async () => {
//     if (input.trim() === '') return;
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);

//     try {
//       const response = await axios.post('http://192.168.11.188:5000/chat', {
//         user_input: input,
//         conversation_history: messages.map((msg) => msg.text).join(" "),  
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
//       <header className="chatbot-header"> 
//         <div className="header-content">
//           <img src={botIcon} alt="Bot-header" className="bot-header" />
//           <h1 className="header-text">Chatbot</h1>
//         </div>
//       </header>
//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div
//             key={index}
//             className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}
//           >
//             {message.sender === 'bot' && (
//               <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             )}
//             <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
//               {message.text}
//             </div>
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








import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './styles/App.css';
import botIcon from './chat.png';

const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

const ChatbotScreen = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('');

  useEffect(() => {
    const savedSessionId = localStorage.getItem('sessionId');
    if (!savedSessionId) {
      const newSessionId = generateSessionId();
      localStorage.setItem('sessionId', newSessionId);
      setSessionId(newSessionId);
    } else {
      setSessionId(savedSessionId);
    }
  }, []);

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const response = await axios.post('http://192.168.11.188:5000/chat', {
        user_input: input,
        conversation_history: messages.map((msg) => msg.text).join("\n"),
        session_id: sessionId,
      });
      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.log("Error:", error);
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
          <h1 className="header-text">Chatbot</h1>
        </div>
      </header>
      <div className="chat-box">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}
          >
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
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default ChatbotScreen;







// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import './styles/App.css';
// import botIcon from './chat.png';

// const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [sessionId, setSessionId] = useState('');
//   const [isFollowUp, setIsFollowUp] = useState(false);

//   useEffect(() => {
//     const savedSessionId = localStorage.getItem('sessionId');
//     if (!savedSessionId) {
//       const newSessionId = generateSessionId();
//       localStorage.setItem('sessionId', newSessionId);
//       setSessionId(newSessionId);
//     } else {
//       setSessionId(savedSessionId);
//     }
//   }, []);

//   const sendMessage = async () => {
//     if (input.trim() === '') return;

//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);

//     try {
//       const response = await axios.post('http://192.168.11.188:5000/chat', {
//         user_input: input,
//         follow_up: isFollowUp,
//         conversation_history: messages.map((msg) => msg.text).join("\n"),
//         session_id: sessionId,
//       });
//       const botMessage = { sender: 'bot', text: response.data.response };
//       setMessages((prevMessages) => [...prevMessages, botMessage]);
//     } catch (error) {
//       console.log("Error:", error);
//       const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
//       setMessages((prevMessages) => [...prevMessages, errorMessage]);
//     }

//     setInput('');
//     setIsFollowUp(false);
//   };

//   const handleKeyPress = (e) => {
//     if (e.key === 'Enter') {
//       sendMessage();
//     }
//   };

//   return (
//     <div className="chat-container">
//       <header className="chatbot-header">
//         <div className="header-content">
//           <img src={botIcon} alt="Bot-header" className="bot-header" />
//           <h1 className="header-text">Chatbot</h1>
//         </div>
//       </header>
//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div
//             key={index}
//             className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}
//           >
//             {message.sender === 'bot' && (
//               <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             )}
//             <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
//               {message.text}
//             </div>
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
//         <button onClick={() => setIsFollowUp(true)}>Follow-up?</button>
//       </div>
//     </div>
//   );
// };

// export default ChatbotScreen;
