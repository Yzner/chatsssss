import React, { useState, useEffect } from "react";
import "../../styles/About.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";

const EmailReq = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);

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
          <h1 className="college-title">EMAIL REQUEST</h1>
        </div>
      </section>

      <section className="Academic-section">
        <div className="academic-container">
          <iframe
            width="560vw"
            height="315vh"
            src="https://docs.google.com/forms/d/e/1FAIpQLSe9QSZF4r4qbbxOp2H19LASV06s6qzA6CItuynDIRzu0Dg1gw/viewform"
            title="Email Request"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          ></iframe>
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

export default EmailReq;
