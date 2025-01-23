
import React, { useState, useEffect } from "react";
import "../styles/Main.css"; 
import ChatbotScreen from "../ChatbotScreen"; 
import botIcon from "../chat.png"; 

const Main = () => {
  const [showChatbot, setShowChatbot] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const toggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };

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

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSu-CS</div>
        <nav>
          <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#organization">Organization</a></li>
            <li><a href="#programs">Degree Programs</a></li>
            <li><a href="#research">Research</a></li>
            <li><a href="#resources">Student Resources</a></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">College of Sciences</h1>
          <p>Palawan State University</p>
          <button className="know-more-btn">Know More</button>
        </div>
      </section>


      <section className="news-events">
        <h3>Welcome to the Official Website of the College of Sciences of Palawan State University.</h3>
        <h2>News and Events</h2>
        <div className="news-items">
          <div className="news-item">
            <img src="news1.jpg" alt="News 1" />
            <p>News Title 1</p>
          </div>
          <div className="news-item">
            <img src="news2.jpg" alt="News 2" />
            <p>News Title 2</p>
          </div>
        </div>
      </section>

      <div className="chatbot-icon" onClick={toggleChatbot}>
        <img src={botIcon} alt="Chatbot Icon" />
      </div>

      {showChatbot && <ChatbotScreen />}

      <footer className="footer">
        <p>© 2025 Palawan State University - College of Science</p>
      </footer>
    </div>
  );
};

export default Main;
