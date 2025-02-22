

import React, { useState, useEffect } from "react";
import "../../styles/About.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  

const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const StudentOrg = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showBubble, setShowBubble] = useState(true); 
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]); 

  const toggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };
  useEffect(() => {
    const interval = setInterval(() => {
      setShowBubble(false);
      setTimeout(() => {
        const randomIndex = Math.floor(Math.random() * chatbotMessages.length);
        setChatbotMessage(chatbotMessages[randomIndex]);
        setShowBubble(true);
      }, 500);
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    if (!window.FB) {
      const script = document.createElement("script");
      script.async = true;
      script.defer = true;
      script.crossOrigin = "anonymous";
      script.src = "https://connect.facebook.net/en_US/sdk.js";
      script.onload = () => {
        window.FB.init({
          xfbml: true,
          version: "v12.0",
        });
      };
      document.body.appendChild(script);
    } else {
      window.FB.XFBML.parse();
    }
  }, []);
  

  const studentOrgs = [
    {
      name: "College Student Government",
      href: "https://www.facebook.com/PSU.CS.CSG",
    },
    {
      name: "Young Biologist Association",
      href: "https://www.facebook.com/psucsyba",
    },
    {
      name: "Association of Computer Scientists",
      href: "https://www.facebook.com/acs.psu",
    },
    {
      name: "Society of Information Technology Enthusiasts",
      href: "https://www.facebook.com/psu.site",
    },
    {
      name: "Environmental Science Student Association",
      href: "https://www.facebook.com/psu.cs.essa",
    },
    {
      name: "Marine Biologists Society",
      href: "https://www.facebook.com/psucsmbs",
    },
  ];

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/about">About</Link></li>
            <li><Link to="/services">Services</Link></li>
            <li><a href="#news">News</a></li>
            <li><a href="#contact">Contact Us</a></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">STUDENT ORGANIZATION</h1>
        </div>
      </section>

      <section className="Page-section">
        <div className="org-grid">
          {studentOrgs.map((org, index) => (
            <div className="org-card" key={index}>
              <h2>{org.name}</h2>
              <a href={org.href} target="_blank" rel="noopener noreferrer">
                <div 
                  className="fb-page"
                  data-href={org.href}
                  data-tabs="timeline"
                  data-width="350"
                  data-height="350"
                  data-small-header="false"
                  data-adapt-container-width="true"
                  data-hide-cover="false"
                  data-show-facepile="true"
                ></div>
              </a>
            </div>
          ))}
        </div>
      </section>


      <div className="chatbot-icon" onClick={toggleChatbot}>
        {showBubble && <div className="chatbot-bubble">{chatbotMessage}</div>}
        <img src={botGif} alt="Chatbot Icon" className="chatbot-gif" />
      </div>

      {showChatbot && (
        <div className="chatbot-container">
          <ChatbotScreen />
        </div>
      )}

      <footer className="footer">
        <p>© 2025 Palawan State University - College of Science</p>
      </footer>
    </div>
  );
};

export default StudentOrg;
