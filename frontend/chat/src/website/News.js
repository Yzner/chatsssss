import React, { useState, useEffect } from "react";
import "../styles/About.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../ChatbotScreen";
import botIcon from "../chat.png";

const News = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
    const [showServicesDropdown, setShowServicesDropdown] = useState(false);

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
    if (!window.FB) {
      const script = document.createElement("script");
      script.async = true;
      script.defer = true;
      script.crossOrigin = "anonymous";
      script.src = "https://connect.facebook.net/en_US/sdk.js#xfbml=1&version=v12.0";
      document.body.appendChild(script);
    }
  }, []);

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li
              className="dropdown"
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
            <li
                className="dropdown"
                onMouseEnter={() => setShowServicesDropdown(true)}
                onMouseLeave={() => setShowServicesDropdown(false)}
            >
                <Link to="/about">Services</Link>
                {showServicesDropdown && (
                <ul className="dropdown-menu">
                    <li><Link to="/MandV">Academic Awards</Link></li>
                    <li><Link to="/GandO">Procedures</Link></li>
                    <li><Link to="/Programs">Enrollment</Link></li>
                    <li><Link to="/CollegeOrgan">Email Request</Link></li>
                </ul>
                )}
            </li>
            <li><a href="#programs">News</a></li>
            <li><a href="#research">Contact Us</a></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">NEWS</h1>
        </div>
      </section>

      <section className="Page-section">
        <h2>COMPEDIUM</h2>
        <div className="Page-container">
          <p>Follow us on Facebook:</p>

          <a 
            href="https://www.facebook.com/compendiumpub" 
            target="_blank" 
            rel="noopener noreferrer"
            style={{ display: "block", textDecoration: "none" }}
          >
            <div 
              className="fb-page" 
              data-href="https://www.facebook.com/compendiumpub" 
              data-tabs="timeline" 
              data-width="500" 
              data-height="500" 
              data-small-header="false" 
              data-adapt-container-width="true" 
              data-hide-cover="false" 
              data-show-facepile="true"
            ></div>
          </a>
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

export default News;
