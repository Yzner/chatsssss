import React, { useState, useEffect } from "react";
import "../../styles/Main.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  
import logoImage from "../Pictures/logocs.png";
const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const AcademicPrograms = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showBubble, setShowBubble] = useState(true); 
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]); 
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

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
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth >= 768) {
        setIsSidebarOpen(false); 
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
              <div className="logo">
                <img src={logoImage} alt="Logo" className="logo-img" />
                PalawanSU-CS
              </div>
              <div class="auth-buttons">
                <a href="/signup" class="get-started-button">Sign Out</a>
              </div>
              {isMobile ? (
                <button className="hamburger" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                  ☰
                </button>
              ) : (
                <nav className="navbars">
                  <ul>
                    <li><Link to="/">Home</Link></li>
                    <li
                      className="dropdown"
                      onMouseEnter={() => setShowAboutDropdown(true)}
                      onMouseLeave={() => setShowAboutDropdown(false)}
                    >
                      <Link to="/about" className="dropdown-toggle">
                        About <span className="arrow">▼</span>
                      </Link>
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
                    >
                      <Link to="/Services">Services</Link>
                    </li>
                    <li><Link to="/News">News</Link></li>
                    <li><Link to="/ContactUs">Contact Us</Link></li>
                  </ul>
                </nav>
              )}
            </header>

      {/* SIDEBAR NAVIGATION */}
      <div className={`sidebar ${isSidebarOpen ? "show" : ""}`}>
        <ul>
          <li><Link to="/">Home</Link></li>
          <li className="dropdown"
            onMouseEnter={() => setShowAboutSideBar(true)}
            onMouseLeave={() => setShowAboutSideBar(false)}
          >
            <Link to="/about">About</Link>
            {showAboutSideBar && (
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
          >
            <Link to="/Services">Services</Link>
          </li>
          <li><Link to="/News">News</Link></li>
          <li><Link to="/ContactUs">Contact Us</Link></li>
        </ul>
      </div>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">ACADEMIC PROGRAMS</h1>
        </div>
      </section>

      <section className="academic-section">
      <p>The prospectus of the different programs of the College of Sciences is shown below.</p>
      
      <div className="academic-row">
        <div className="drive-container">
          <iframe
            src="https://drive.google.com/embeddedfolderview?id=14_M8J60_cuYa_uPIO1Y-JzQK3ZSBdvRJ#grid"
            style={{ width: "100%", height: "400px", border: "none" }}
            title="Academic Programs Drive 1"
          ></iframe>
        </div>
        <div className="drive-container">
          <iframe
            src="https://drive.google.com/embeddedfolderview?id=14_M8J60_cuYa_uPIO1Y-JzQK3ZSBdvRJ#grid"
            style={{ width: "100%", height: "400px", border: "none" }}
            title="Academic Programs Drive 2"
          ></iframe>
        </div>
        <div className="drive-container">
          <iframe
            src="https://drive.google.com/embeddedfolderview?id=14_M8J60_cuYa_uPIO1Y-JzQK3ZSBdvRJ#grid"
            style={{ width: "100%", height: "400px", border: "none" }}
            title="Academic Programs Drive 3"
          ></iframe>
        </div>
        <div className="drive-container">
          <iframe
            src="https://drive.google.com/embeddedfolderview?id=14_M8J60_cuYa_uPIO1Y-JzQK3ZSBdvRJ#grid"
            style={{ width: "100%", height: "400px", border: "none" }}
            title="Academic Programs Drive 4"
          ></iframe>
        </div>
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
    </div>
  );
};

export default AcademicPrograms;
