
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "../styles/Main.css";
import ChatbotScreen from "../ChatbotScreen";
import botGif from "./Pictures/CHAT.gif"; 
import featureImage from "./Pictures/faculty.png";
import featureImage1 from "./Pictures/email.PNG";
import featureImage2 from "./Pictures/accup.png";
import featureImage3 from "./Pictures/message.png";
import featureImage4 from "./Pictures/news.png";

const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const Main = () => {
  const [showChatbot, setShowChatbot] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);
  const [showServicesSideBar, setShowServicesSideBar] = useState(false);
  const [showBubble, setShowBubble] = useState(true);
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

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
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };
  
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
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

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        {isMobile ? (
          <button className="hamburger" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            ☰
          </button>
        ) : (
          <nav>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li className="dropdown"
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
              <li className="dropdown"
                onMouseEnter={() => setShowServicesDropdown(true)}
                onMouseLeave={() => setShowServicesDropdown(false)}
              >
                <Link to="/Services">Services</Link>
                {showServicesDropdown && (
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

      <section className="welcome-sec">
        <div className="welcome-text">
          <h1 className="college-title">College of Sciences</h1>
          <p>Palawan State University</p>
          <Link to="/about" className="feature-link">Know More</Link>
        </div>
      </section>

      <section className="news-events">
        <h3>Welcome to the Official Website of the College of Sciences of Palawan State University.</h3>
      </section>

      {/* Feature Sections */}
      {[{
        image: featureImage, title: "COLLEGE FACULTY & STAFF", 
        text: "To help everyone get to know our College Instructors and Staff, click the link to view our current roster.", 
        link: "/CollegeOrgan"
      },{
        image: featureImage1, title: "EMAIL REQUEST TOOL",
        text: "Click here to request for your PSU Email Address. Applicable to those who do not yet have their email address.",
        link: "/EmailReq"
      },{
        image: featureImage2, title: "ACCREDITATION PORTAL",
        text: "The College of Sciences is undergoing the AACCUP Online Program Accreditation. Click to visit (Limited Access).",
        link: "/Accreditation"
      },{
        image: featureImage3, title: "WELCOME MESSAGE",
        text: "Welcome to the College of Sciences! Watch Prof. Imelda R. Lactuan, Dean of the College of Sciences, as she shares her message.",
        link: "/WelcomeMessage"
      },{
        image: featureImage4, title: "NEWS AND ANNOUNCEMENTS",
        text: "News Items are now published. Student News are handled by the Compendium but will be linked here.",
        link: "/News"
      }].map(({ image, title, text, link }, index) => (

        <section
          key={index}
          className={`feature-section ${isMobile ? "center-layout" : "left-layout"}`}
        >
          <div className="feature-container">
            <img src={image} alt={title} className="feature-image" />
            <div className="feature-content">
              <h2>{title}</h2>
              <p>{text}</p>
              <Link to={link} className="feature-link">Click Here</Link>
            </div>
          </div>
        </section>
      ))}

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

export default Main;

