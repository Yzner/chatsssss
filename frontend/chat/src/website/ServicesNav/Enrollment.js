import React, { useState, useEffect } from "react";
import "../../styles/Main.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";

const Enrollment = () => {
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
          <h1 className="college-title">ENROLLMENT</h1>
        </div>
      </section>

      <section>
      <h3>Welcome to the College of Sciences Enrollment Portal. We took the effort to put all these resources in one place, for your ease of access. If you have any questions, please do not hesitate to contact your Faculty-Adviser / Faculty-Enrolling Officer. </h3>
      <h3>Note: Our enrollment is face-to-face. Please observe all health and safety protocols at all times. Thank you so much!!!</h3>
        <h2>ENROLLMENT SCHEDULE</h2>
      </section>

      <div className="chatbot-icon" onClick={toggleChatbot}>
        <img src={botIcon} alt="Chatbot Icon" />
      </div>

      {showChatbot && <ChatbotScreen />}


    </div>
  );
};

export default Enrollment;
