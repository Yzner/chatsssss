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
const Dropping = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
  const [showServicesSideBar, setShowServicesSideBar] = useState(false);
  const [showBubble, setShowBubble] = useState(true);
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]);
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
            onMouseEnter={() => setShowServicesSideBar(true)}
            onMouseLeave={() => setShowServicesSideBar(false)}
          >
            <Link to="/Services">Services</Link>
            {showServicesSideBar && (
              <ul className="dropdown-menu">
                <li><Link to="/AcadAwards">Academic Awards</Link></li>
                <li><Link to="/Procedures">Procedures</Link></li>
                <li><Link to="/Enrollment">Enrollment</Link></li>
                <li><Link to="/EmailReq">Email Request</Link></li>
              </ul>
            )}
          </li>
          <li><Link to="/News">News</Link></li>
          <li><Link to="/ContactUs">Contact Us</Link></li>
        </ul>
      </div>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">DROPPING</h1>
        </div>
      </section>

      <section className="content-section">
      <div className="shifting-container">
      <header className="shifting-header">
        {/* <img src={logo} alt="College of Sciences Logo" className="college-logo" /> */}
        <h1>Dropping Procedure</h1>
      </header>
      <div className="shifting-steps">
        <div class="step-box">
          <div class="step-header">
              <span class="step-number">01.</span> Online Portal
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> Students shall access their portal through https://ourpsu.com/ and once logged-in, they shall click the adding/changing/dropping button. 
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">02.</span> Dropping of subject/s
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> The student shall select the courses they need to drop then click submit.
              <p>Note: The student is required to explain their reason for dropping of their subject/s</p>
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">03.</span> Review
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> The program chairperson shall review the submitted requests for dropping of subjects and once confirmed they will click the “recommending approval” button. If disapproved, they shall click the “disapproved” button then click “send email” and type in the reason.
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">04.</span> Subject/s recommended
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> Once the request for dropping of subjects has been recommended by the program chairperson, the associate dean shall check and click the approved/disapproved button.
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">05.</span> Checking for Payment/Not
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> The University Registrar shall check all approved requests and determine if the request requires payment or not. If the said request requires payment, please proceed to step 6. If it does not require payment, please proceed to step 7.
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">06.</span> Payment
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> The student shall proceed to the Cashier’s Office for payment of their dropping of subjects.
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">07.</span> Validating
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> Once paid, the accounting office shall validate the said fees.
          </div>
        </div>

        <div class="step-box">
          <div class="step-header">
              <span class="step-number">08.</span> Validated
          </div>
          <div class="step-content">
              <span class="bullet">✳</span> After validation, the updated COR will be sent to the students through their corporate email address. The students can also access their COR by logging-in to their OECAS account.
          </div>
        </div>
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

export default Dropping;
