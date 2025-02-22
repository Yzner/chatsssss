
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "../styles/Main.css";
import ChatbotScreen from "../ChatbotScreen";
import botGif from "./Pictures/CHAT.gif"; 

const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const Main = () => {
  const [showChatbot, setShowChatbot] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);
  const [showBubble, setShowBubble] = useState(true);
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]); 

  const toggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

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

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li className="dropdown"
              onMouseEnter={() => setShowAboutDropdown(true)}
              onMouseLeave={() => setShowAboutDropdown(false)}
            >
              <Link to="/about">About</Link>
              {showAboutDropdown && (
                <ul className="dropdown-menu">
                  <li><Link to="/MandV">University Mission & Vision</Link></li>
                  <li><Link to="/GandO">College Goals and Objectives</Link></li>
                  <li><Link to="/Programs">Academic Programs</Link></li>
                  <li><Link to="/CollegeOrgan">Faculty & Staff</Link></li>
                  <li><Link to="/StudentOrg">College Student Organizations</Link></li>
                </ul>
              )}
            </li>
            <li className="dropdown"
              onMouseEnter={() => setShowServicesDropdown(true)}
              onMouseLeave={() => setShowServicesDropdown(false)}
            >
              <Link to="/Services">Services</Link>
              {showServicesDropdown && (
                <ul className="dropdown-menu">
                  <li><Link to="/AcadAwards">Academic Awards</Link></li>
                  <li><Link to="/Procedures">Procedures</Link></li>
                  <li><Link to="/Enrollment">Enrollment</Link></li>
                  <li><Link to="/EmailReq">Email Request</Link></li>
                </ul>
              )}
            </li>
            <li><Link to="/News">News</Link></li>
            <li><Link to="/ContactUs">Contact Us</Link></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-sec">
        <div className="welcome-text">
          <h1 className="college-title">College of Sciences</h1>
          <p>Palawan State University</p>
          <button className="know-more-btn">Know More</button>
        </div>
      </section>

      <section className="news-events">
        <h3>Welcome to the Official Website of the College of Sciences of Palawan State University.</h3>
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
    </div>
  );
};

export default Main;
