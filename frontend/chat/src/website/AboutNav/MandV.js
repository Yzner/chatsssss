import React, { useState, useEffect } from "react";
import "../../styles/About.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";
import { FaChevronDown } from "react-icons/fa";  // Import arrow icon

const MissionVision = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showGoalContent, setShowGoalContent] = useState(false); // Toggle goal section

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/about">About</Link></li>
            <li><Link to="#news">News</Link></li>
            <li><Link to="#contact">Contact Us</Link></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">UNIVERSITY MISSION AND VISION</h1>
        </div>
      </section>

      <section className="content-section">
        <div className="columns-container">
          <div className="column-item">
            <h3>Vision</h3>
            <p>An internationally recognized university that provides relevant and innovative education and research for lifelong learning and sustainable development.</p>
          </div>
          <div className="column-item">
            <h3>Mission</h3>
            <p>The Palawan State University is committed to upgrading the quality of life of the people by providing higher education opportunities.</p>
          </div>
          <div className="column-item">
            <h3>Core Values</h3>
            <p>EQUALITY: Excellence, Quality Assurance, Unity, Leadership, Innovation, Transparency, and Youth Empowerment.</p>
          </div>
        </div>
      </section>


      <div className="chatbot-icon" onClick={() => setShowChatbot(!showChatbot)}>
        <img src={botIcon} alt="Chatbot Icon" />
      </div>
      {showChatbot && <ChatbotScreen />}

      <footer className="footer">
        <p>© 2025 Palawan State University - College of Science</p>
      </footer>
    </div>
  );
};

export default MissionVision;
