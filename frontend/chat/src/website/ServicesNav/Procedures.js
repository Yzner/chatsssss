import React, { useState, useEffect } from "react";
import "../../styles/About.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";

const Procedures = () => {
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
          <h1 className="college-title">PROCEDURES</h1>
        </div>
      </section>

      <section>
      <h3>The following are some of the common procedures under the College of Sciences</h3>
        <h2>To find out more about the College, click here: </h2>
        <ul>
            <li><Link to ="/Shifting">Shifting</Link></li>
            <li><Link to="/GenClean">General Clearance</Link></li>
            <li><Link to="/GoodMo">Good Moral Request</Link></li>
            <li><Link to="/Comple">Completion/Removal</Link></li>
            <li><Link to="/Substitute">Substitution/Credit</Link></li>
            <li><Link to="/CrossEnroll">Cross-Enrollment</Link></li>
            <li><Link to="/Shift">Acceptance to Shift</Link></li>
            <li><Link to="/Adding">Adding</Link></li>
            <li><Link to="/Changing">Changing</Link></li>
            <li><Link to="/Dropping">Dropping</Link></li>
        </ul>
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

export default Procedures;
