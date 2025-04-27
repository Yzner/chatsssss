
// import React, { useState, useEffect } from 'react';
// import axios from 'axios';
// import './styles/App.css';
// import botIcon from "./website/Pictures/logocs.png";
// import { IoSend } from 'react-icons/io5';



// const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [sessionId, setSessionId] = useState('');
//   const [isTyping, setIsTyping] = useState(false);


//   // useEffect(() => {
//   //   const savedSessionId = localStorage.getItem('sessionId');
//   //   const userId = localStorage.getItem('userId');

//   //   if (!savedSessionId) {
//   //     const newSessionId = generateSessionId();
//   //     localStorage.setItem('sessionId', newSessionId);
//   //     setSessionId(newSessionId);
//   //   } else {
//   //     setSessionId(savedSessionId);
//   //   }

//   //   const fetchConversationHistory = async () => {
//   //     try {
//   //       const response = await axios.get('http://localhost:5000/conversation_history', {
//   //         params: { session_id: savedSessionId || sessionId, user_id: userId },
//   //       });
        
//   //       console.log("Fetched conversation history:", response.data);

//   //       if (Array.isArray(response.data)) {
//   //         setMessages(response.data.map((msg) => ({
//   //           sender: msg.sender,
//   //           text: msg.message,
//   //         })));
//   //       }
//   //     } catch (error) {
//   //       console.error("Error fetching conversation history:", error);
//   //     }
//   //   };

//   //   fetchConversationHistory();
//   // }, [sessionId]);

//   useEffect(() => {
//     const savedSessionId = localStorage.getItem('sessionId');
//     const userId = localStorage.getItem('userId');
  
//     if (!savedSessionId) {
//       const newSessionId = generateSessionId();
//       localStorage.setItem('sessionId', newSessionId);
//       setSessionId(newSessionId);
//     } else {
//       setSessionId(savedSessionId);
//     }
  
//     const fetchConversationHistory = async () => {
//       try {
//         const response = await axios.get('http://localhost:5000/conversation_history', {
//           params: { session_id: savedSessionId || sessionId, user_id: userId },
//         });
  
//         console.log("Fetched conversation history:", response.data);
  
//         if (Array.isArray(response.data) && response.data.length > 0) {
//           setMessages(response.data.map((msg) => ({
//             sender: msg.sender,
//             text: msg.message,
//           })));
//         } else {
//           setMessages([{ sender: 'bot', text: "Welcome to Ask.CS! How can I help you today?" }]);
//         }
//       } catch (error) {
//         console.error("Error fetching conversation history:", error);
//         setMessages([{ sender: 'bot', text: "Welcome to Ask.CS! How can I help you today?" }]);
//       }
//     };
  
//     fetchConversationHistory();
//   }, [sessionId]);
  

//   // const sendMessage = async () => {
//   //   if (input.trim() === '') return;

//   //   const userMessage = { sender: 'user', text: input };
//   //   setMessages((prevMessages) => [...prevMessages, userMessage]);

//   //   try {
//   //     const userId = localStorage.getItem('userId');
//   //     const response = await axios.post('http://localhost:5000/chat', {
//   //       user_input: input,
//   //       session_id: sessionId,
//   //       user_id: userId,
//   //     });

//   //     console.log("Bot response:", response.data);
//   //     const botMessage = { sender: 'bot', text: response.data.response };
//   //     setMessages((prevMessages) => [...prevMessages, botMessage]);
//   //   } catch (error) {
//   //     console.error("Error:", error);
//   //     const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
//   //     setMessages((prevMessages) => [...prevMessages, errorMessage]);
//   //   }

//   //   setInput('');
//   // };


//   const sendMessage = async () => {
//     if (input.trim() === '') return;
  
//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);
//     setInput('');
//     setIsTyping(true);
  
//     try {
//       const userId = localStorage.getItem('userId');
//       const response = await axios.post('http://localhost:5000/chat', {
//         user_input: input,
//         session_id: sessionId,
//         user_id: userId,
//       });
  
//       const fullText = response.data.response;
//       let displayedText = '';
//       let i = 0;
  
//       const typeOut = () => {
//         if (i < fullText.length) {
//           displayedText += fullText[i];
//           setMessages((prevMessages) => [
//             ...prevMessages.slice(0, -1), 
//             { sender: 'bot', text: displayedText }
//           ]);
//           i++;
//           setTimeout(typeOut, 900);
//         } else {
//           setIsTyping(false);
//         }
//       };
//       setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: '' }]);
//       setTimeout(typeOut, 20000); 
//     } catch (error) {
//       console.error("Error:", error);
//       const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
//       setMessages((prevMessages) => [...prevMessages, errorMessage]);
//       setIsTyping(false);
//     }
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
//           <h1 className="header-text">Ask.CS</h1>
//         </div>
//       </header>
//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div key={index} className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}>
//             {message.sender === 'bot' && (
//               <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             )}
//             <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
//               {message.text}
//             </div>
//           </div>
//         ))}
//         {isTyping && (
//           <div className="message-row bot-row">
//             <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             <div className="message bot-message typing-indicator">
//               <span className="dot"></span>
//               <span className="dot"></span>
//               <span className="dot"></span>
//             </div>
//           </div>
//         )}
//       </div>

//       <div className="input-box">
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyPress={handleKeyPress}
//           placeholder="Type your question..."
//         />
//         <button onClick={sendMessage} className="send-button">
//          <IoSend size={24} color="green" />
//         </button>

//       </div>
//     </div>
//   );
// };

// export default ChatbotScreen;






// gawa-gawa part2 chatbotscreen.js

// import React, { useState, useEffect, useRef } from 'react';
// import axios from 'axios';
// import './styles/App.css';
// import botIcon from "./website/Pictures/logocs.png";
// import { IoSend } from 'react-icons/io5';



// const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

// const ChatbotScreen = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [sessionId, setSessionId] = useState('');
//   const messagesEndRef = useRef(null);

//   // Instant scroll to bottom
//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: 'auto' }); // No animation
//   };

//   useEffect(() => {
//     const savedSessionId = localStorage.getItem('sessionId');
//     const userId = localStorage.getItem('userId');

//     if (!savedSessionId) {
//       const newSessionId = generateSessionId();
//       localStorage.setItem('sessionId', newSessionId);
//       setSessionId(newSessionId);
//     } else {
//       setSessionId(savedSessionId);
//     }

//     const fetchConversationHistory = async () => {
//       try {
//         const response = await axios.get('http://localhost:5000/conversation_history', {
//           params: { session_id: savedSessionId || sessionId, user_id: userId },
//         });

//         console.log("Fetched conversation history:", response.data);

//         if (Array.isArray(response.data)) {
//           const formattedMessages = response.data.map((msg) => ({
//             sender: msg.sender,
//             text: msg.message,
//           }));
//           setMessages(formattedMessages);

//           // Delay scroll to ensure messages rendered
//           setTimeout(() => {
//             scrollToBottom(); // Jump straight to latest convo
//           }, 0);
//         }
//       } catch (error) {
//         console.error("Error fetching conversation history:", error);
//       }
//     };

//     fetchConversationHistory();
//   }, [sessionId]);

//   const sendMessage = async () => {
//     if (input.trim() === '') return;

//     const userMessage = { sender: 'user', text: input };
//     setMessages((prevMessages) => [...prevMessages, userMessage]);

//     try {
//       const userId = localStorage.getItem('userId');
//       const response = await axios.post('http://localhost:5000/chat', {
//         user_input: input,
//         session_id: sessionId,
//         user_id: userId,
//       });

//       console.log("Bot response:", response.data);
//       const botMessage = { sender: 'bot', text: response.data.response };
//       setMessages((prevMessages) => [...prevMessages, botMessage]);
//     } catch (error) {
//       console.error("Error:", error);
//       const errorMessage = { sender: 'bot', text: "Oops! Something went wrong." };
//       setMessages((prevMessages) => [...prevMessages, errorMessage]);
//     }

//     setInput('');
//   };

//   // Smooth scroll on message update (optional)
//   useEffect(() => {
//     if (messages.length > 0) {
//       messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
//     }
//   }, [messages]);

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
//           <h1 className="header-text">Ask.CS</h1>
//         </div>
//       </header>

//       <div className="chat-box">
//         {messages.map((message, index) => (
//           <div key={index} className={`message-row ${message.sender === 'user' ? 'user-row' : 'bot-row'}`}>
//             {message.sender === 'bot' && (
//               <img src={botIcon} alt="Bot Icon" className="bot-icon" />
//             )}
//             <div className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}>
//               {message.text}
//             </div>
//           </div>
//         ))}
//         <div ref={messagesEndRef} />
//       </div>

//       <div className="input-box">
//         <input
//           type="text"
//           value={input}
//           onChange={(e) => setInput(e.target.value)}
//           onKeyPress={handleKeyPress}
//           placeholder="Type your question..."
//         />
//         <button onClick={sendMessage} className="send-button">
//          <IoSend size={24} color="green" />
//         </button>

//       </div>
//     </div>
//   );
// };

// export default ChatbotScreen;





import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './styles/App.css';
import botIcon from "./website/Pictures/logocs.png";
import { IoSend } from 'react-icons/io5';

const generateSessionId = () => 'session_' + Math.random().toString(36).substr(2, 9);

const ChatbotScreen = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom(); // Scroll when messages change
  }, [messages]);

  useEffect(() => {
    const savedSessionId = localStorage.getItem('sessionId');
    const userId = localStorage.getItem('userId');

    const newSessionId = savedSessionId || generateSessionId();
    if (!savedSessionId) localStorage.setItem('sessionId', newSessionId);
    setSessionId(newSessionId);

    const fetchConversationHistory = async () => {
      try {
        const response = await axios.get('http://localhost:5000/conversation_history', {
          params: { session_id: newSessionId, user_id: userId },
        });

        if (Array.isArray(response.data) && response.data.length > 0) {
          setMessages(response.data.map(msg => ({
            sender: msg.sender,
            text: msg.message,
          })));
        } else {
          setMessages([{ sender: 'bot', text: "Welcome to Ask.CS! How can I help you today?" }]);
        }
      } catch (error) {
        console.error("Error fetching conversation history:", error);
        setMessages([{ sender: 'bot', text: "Welcome to Ask.CS! How can I help you today?" }]);
      }
    };

    fetchConversationHistory();
  }, []);

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const userId = localStorage.getItem('userId');
      const response = await axios.post('http://localhost:5000/chat', {
        user_input: input,
        session_id: sessionId,
        user_id: userId,
      });

      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
      setMessages(prev => [...prev, { sender: 'bot', text: "Oops! Something went wrong." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') sendMessage();
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

        {isTyping && (
          <div className="message-row bot-row">
            <img src={botIcon} alt="Bot Icon" className="bot-icon" />
            <div className="message bot-message typing-indicator">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}

        {/* Ref target for scrolling */}
        <div ref={messagesEndRef} />
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
