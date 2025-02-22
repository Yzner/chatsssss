import React, { useState, useEffect } from "react";
import "../../styles/About.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";

const AcadAwards = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [expandedSection, setExpandedSection] = useState(null);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);

  const toggleChatbot = () => {
    setShowChatbot(!showChatbot);
  };

  const toggleSection = (section) => {
    setExpandedSection((prevSection) => (prevSection === section ? null : section));
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
          <h1 className="college-title">ACADEMIC AWARDS</h1>
        </div>
      </section>

      <section className="AwardsApp-section">
      <div className="expandable-section">
        <div>
            <h3>Please take note of some important reminders:</h3>
            <p>1. GWA should be 1.75 or higher</p>
            <p>2. NSTP is excluded in the computation</p>
            <p>3. Only Regular Students may apply</p>
            <p>4. Must have no “INC” / “No Grade” remarks</p>
            <p>For issues or concerns, please email cs@psu.palawan.edu.ph</p>     
        </div>
        </div>
      </section>

      <div className="chatbot-icon" onClick={toggleChatbot}>
        <img src={botIcon} alt="Chatbot Icon" />
      </div>

      {showChatbot && <ChatbotScreen />}
    </div>
  );
};

export default AcadAwards;
