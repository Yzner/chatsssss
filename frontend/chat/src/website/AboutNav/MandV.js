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

const MissionVision = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showBubble, setShowBubble] = useState(true); 
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]); 
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
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
      setScrolled(window.scrollY > 50);
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
          <h1 className="college-title">UNIVERSITY MISSION AND VISION</h1>
        </div>
      </section>

      <section className="content-section">
        <div className="columns-container">
          <div className="column-item">
            <h3>VISION</h3>
            <p>An internationally recognized university that provides relevant and innovative education and research for lifelong learning and sustainable development.</p>
            <h3>MISSION</h3>
            <p>Palawan State University is committed to upgrade people's quality of life by providing education opportunities through excellent instruction, research and innovation, extension, production services, and transnational collaborations.</p>
          </div>
          <div className="column-item">
            <h3>CORE VALUES</h3>
            <p>Excellence in service, Quality assurance, Unity in diversity, Advocacy for sustainable development, Leadership by example, Innovation, Transparency, and Youth empowerment (EQUALITY).</p>
          </div>
          <div className="column-item">
            <h3>QUALITY POLICY</h3>
            <p>"We Provide equal opportunities for relevant, innovative, and internationally recognized higher education programs and advanced studies for lifelong learning and sustainable development.</p> 
            <p>We Strongly commit to deliver excellence in instruction, research, extension, and transnational programs in order to meet the increasing levels of stakeholder demand as well as statutory and regulatory requirements.</p> 
            <p>The University shall continually monitor, review and upgrade its quality management system to ensure compliance with national and international standards and requirements." </p>
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

export default MissionVision;
