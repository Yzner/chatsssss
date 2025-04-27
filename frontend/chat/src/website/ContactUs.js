import React, { useState, useEffect } from "react";
import "../styles/Main.css"; 
import { Link } from "react-router-dom"; 
import ChatbotScreen from "../ChatbotScreen";
import botGif from "./Pictures/CHAT.gif"; 
import logoImage from "./Pictures/logocs.png";
const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const ContactUs = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
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
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

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
      if (window.innerWidth >= 768) {
        setIsSidebarOpen(false); 
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  
  const studentOrgs = [
    {
      name: "College of Sciences - College Student Government",
      href: "https://www.facebook.com/PSU.CS.CSG",
    },
    {
      name: "Palawan State University - College of Sciences ",
      href: "https://www.facebook.com/profile.php?id=61557203510612",
    },
  ];

  useEffect(() => {
    const loadFacebookSDK = () => {
      if (!window.FB) {
        const script = document.createElement("script");
        script.async = true;
        script.defer = true;
        script.crossOrigin = "anonymous";
        script.src = "https://connect.facebook.net/en_US/sdk.js";
        script.onload = () => {
          window.FB.init({
            xfbml: true,
            version: "v12.0",
          });
          window.FB.XFBML.parse();
        };
        document.body.appendChild(script);
      } else {
        window.FB.XFBML.parse();
      }
    };
  
    loadFacebookSDK();
  }, []);
  
  // This ensures it re-parses after studentOrgs renders
  useEffect(() => {
    if (window.FB) {
      window.FB.XFBML.parse();
    }
  }, [studentOrgs]);
  

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
          <h1 className="college-title">Contact Us</h1>
        </div>
      </section>

      <section className="contact-section">
      <div className="contact-header">
        <h1>For questions or inquiries:</h1>
        <p>Please feel free to contact us through the following channels / medium</p>
      </div>
      <div className="contact-grid">
        {studentOrgs.map((org, index) => (
          <div className="contact-card" key={index}>
            <div className="contact-left">
            <div
              id={`fb-page-${index}`}
              className="fb-page"
              data-href={org.href}
              data-tabs="timeline"
              data-width="350"
              data-height="350"
              data-small-header="false"
              data-adapt-container-width="true"
              data-hide-cover="false"
              data-show-facepile="true"
            ></div>
            </div>
            <div className="contact-right">
              <h2>{org.name}</h2>
              <p>
                Stay updated by following the official Facebook page of the {org.name}. Get the latest announcements, events, and student activities here!
              </p>
            </div>
          </div>
        ))}

        {/* Google Map Card */}
        <div className="contact-card">
          <div className="contact-left">
            <iframe
              title="PSU CS Map"
              src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d125537.99293933005!2d118.67553631249435!3d9.757698099999996!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x33b5607e45f2f20f%3A0x84d0f4761a0c3d5b!2sPalawan%20State%20University!5e0!3m2!1sen!2sph!4v1713538820001!5m2!1sen!2sph"
              width="350"
              height="350"
              style={{ border: 0 }}
              allowFullScreen=""
              loading="lazy"
              referrerPolicy="no-referrer-when-downgrade"
            ></iframe>
          </div>
          <div className="contact-right">
            <h2>Palawan State University - College of Sciences Location</h2>
            <p>
              Visit us in person! Find the College of Sciences on the map and plan your visit to our campus in Puerto Princesa, Palawan.
            </p>
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

export default ContactUs;
