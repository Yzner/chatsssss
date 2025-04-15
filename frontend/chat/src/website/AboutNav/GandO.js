import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FaChevronDown } from "react-icons/fa"; 
import "../../styles/Main.css"; 
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  
import logoImage from "../Pictures/logocs.png";
const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const goalsData = [
  {
    id: "collegeGoals",
    title: "GOAL OF THE COLLEGE OF SCIENCES",
    content: "To advance the frontiers of knowledge in the environmental, computer, mathematical, and biological sciences and to be a world-class center of learning, scientific inquiry, and research beneficial to the general welfare of the People."
  },
  {
    id: "CScollegeGoals",
    title: "BS COMPUTER SCIENCE - PROGRAM GOALS",
    content: `
      **Program Outcomes - Computer Science:**
      \n1. Design appropriate computing solutions.
      \n2. Apply computing knowledge to real-world problems.
      \n3. Utilize modern computing tools and methodologies.
      \n4. Engage in lifelong learning.
    `
  },
  {
    id: "EScollegeGoals",
    title: "BS ENVIRONMENTAL SCIENCE - PROGRAM GOALS",
    content: `
      **Program Outcomes - Environmental Science:**
      \n1. Understand environmental sustainability.
      \n2. Conduct research on environmental conservation.
      \n3. Promote responsible environmental practices.
    `
  },
  {
    id: "ITcollegeGoals",
    title: "BS INFORMATION TECHNOLOGY - PROGRAM GOALS",
    content: `
      **Program Outcomes - IT Department:**
      \n1. Develop and manage IT solutions.
      \n2. Ensure cybersecurity and data integrity.
      \n3. Innovate solutions for IT-related challenges.
    `
  }
];

const GoalsObjective = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [expandedSection, setExpandedSection] = useState(null);

  const toggleSection = (id) => setExpandedSection(expandedSection === id ? null : id);
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
    const handleScroll = () => setScrolled(window.scrollY > 50);
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
          <h1 className="college-title">GOALS AND OBJECTIVES</h1>
        </div>
      </section>

      <section className="goal-section">
        {goalsData.map(({ id, title, content }) => (
          <div key={id} className="expandable-section">
            <div className="goal-header" onClick={() => toggleSection(id)}>
              <h1>{title}</h1>
              <FaChevronDown className={`arrow-icon ${expandedSection === id ? "rotated" : ""}`} />
            </div>
            {expandedSection === id && <p className="goal-content">{content}</p>}
          </div>
        ))}
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

export default GoalsObjective;
