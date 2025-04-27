
import React, { useState, useEffect } from "react";
import "../../styles/Main.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  
import onaphoto from "../Pictures/ona.JPG";
import royphoto from "../Pictures/roy.jpg";
import renephoto from "../Pictures/rene.JPG";
import kristinephoto from "../Pictures/kristine.jpg";
import menchphoto from "../Pictures/mench.JPG";
import ivanImage from "../Pictures/ivan.JPG";
import logoImage from "../Pictures/logocs.png";
import florphoto from "../Pictures/flor.jpg";
import rizzaphoto from "../Pictures/rizza.JPG";
import kimphoto from "../Pictures/kim.JPG";
import bemphoto from "../Pictures/bem.JPG";
import regphoto from "../Pictures/reg.JPG";
import demyphoto from "../Pictures/demy.JPG";
import divphoto from "../Pictures/div.JPG";
import jenphoto from "../Pictures/jen.JPG";
import maryphoto from "../Pictures/mary.jpg";
import ericphoto from "../Pictures/eric.JPG";
import adonImage from "../Pictures/adon.png";
import jandiImage from "../Pictures/jandi.png";
import myrnaphoto from "../Pictures/myrna.JPG";
import doctophoto from "../Pictures/docto.jpg";
import vpphoto from "../Pictures/vp.jpg";


const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const facultyMembers = [
  {
    name: "DR. RAMON M. DOCTO",
    title: "UNIVERSITY PRESIDENT",
    photo: doctophoto,
  },
  {
    name: "DR. MAILA N. LUCERO",
    title: "VP FOR ACADEMIC AFFAIRS", 
    photo: vpphoto,
  },
  {
    name: "DR. RONALD EDILBERTO A. ONA",
    title: "DEAN, COLLEGE OF SCIENCES",
    photo: onaphoto,
  },
  {
    name: "MR. JANDI PANOLINO",
    title: "ASSOCIATE DEAN",
    photo: jandiImage,
  },
  {
    name: "MS. MYRNA L. DIVEDOR",
    title: "STAFF, DEANS OFFICE",
    photo: myrnaphoto,
  },
  {
    name: "MR. BRENDAN R. SARRA",
    title: "STAFF, CS STOCKROOM",
    photo: logoImage,
  },
  {
    name: "MR. RENE V. BULIAG",
    title: "CHAIRPERSON, COMPUTER STUDIES DEPARTMENT",
    photo: renephoto,
  },
  {
    name: "KRISTINE JOY V. MARTINEZ",
    title: "FACULTY",
    photo: kristinephoto,
  },
  {
    name: "MENCHIE LOPEZ",
    title: "FACULTY",
    photo:menchphoto,
  },
  {
    name: "DEMY R. DIZON",
    title: "FACULTY",
    photo: demyphoto,
  },
  {
    name: "FLOREDITH JEANNE G. ALCID ",
    title: "FACULTY",
    photo: florphoto,
  },
  {
    name: "MARIA RIZZA LAUDIANO-ARMILDEZ",
    title: "FACULTY",
    
    photo: rizzaphoto,
  },
  {
    name: "KIM C. BAGUIO",
    title: "FACULTY",
    
    photo: kimphoto,
  },
  {
    name: "MA. REGINA P. BRAVO",
    title: "FACULTY",
    
    photo: regphoto,
  },
  {
    name: "BEMSOR T. CAABAY",
    title: "FACULTY",
    
    photo: bemphoto,
  },
  {
    name: "DIVINE P. CAABAY",
    title: "FACULTY",
    
    photo: divphoto,
  },
  {
    name: "JENNIFER G. RABANG",
    title: "FACULTY",
    
    photo: jenphoto,
  },
  {
    name: "ALBERT IVAN R. CASTILLO",
    title: "FACULTY",
    
    photo: ivanImage,
  },
  {
    name: "MARY JOY DELOS TRINOS",
    title: "FACULTY",
    
    photo: maryphoto,
  },
  {
    name: "TIFFANY O. PABILONA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "AILENE L. SIBAYAN-BOBIER",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ERIC HENRY P. RIVERA",
    title: "FACULTY",
    
    photo: ericphoto,
  },
  {
    name: "ADONIS C. AMPONGAN",
    title: "FACULTY",
    
    photo: adonImage,
  },
  {
    name: "ROY CYRIL D. DOSADO",
    title: "FACULTY",
    
    photo: royphoto,
  },
  //MATH DEPARTMENT
  {
    name: "DR. MARICHU G. MOZO",
    title: "CHAIRPERSON, MATHEMATICS DEPARTMENT",
    
    photo: logoImage,
  },
  {
    name: "PROF. MA. CONCEPCION G. ABIAN",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MS. JESSA MAE P. ABRINA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "PROF. JULIANA B. ARQUERO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MS. BLESSY ANN L. AYCO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "PROF. MARY JANE A. BUNDAC",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MS. MARY JOYCE C. DUEÑAS",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MR. REUBEN YSMAEL A. GANAPIN",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MR. LELANDO H. GATON",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "PROF. JERICO M. PADRONES",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "PROF. SIOTE I. WY",
    title: "FACULTY",
    
    photo: logoImage,
  },
  //BIO-PHYSICAL SCIENCE DEPARTMENT
  {
    name: "IMELDA C. PACALDO",
    title: "CHAIRPERSON, BIO-PHYSICAL SCIENCE DEPARTMENT",
    
    photo: logoImage,
  },
  {
    name: "JANETH S. DAGANTA",
    title: "PROGRAM HEAD- MEDICAL BIOLOGY",
    
    photo: logoImage,
  },
  {
    name: "RENALYN O. SEGUERRA ",
    title: "PROGRAM HEAD- MARINE BIOLOGY",
    
    photo: logoImage,
  },
  {
    name: "RODOLFO JR. O. ABALUS",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "GERONIMO ALLAN JEROME G. ACOSTA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "AZAREEL AMON O. ALVAREZ",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "RYAN N. ARANGA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "JAYBIE S. ARZAGA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ARSELENE U. BITARA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "PRINCESS EUNICE C. DENOSTA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "JEAN MARIE L. DIEGO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "FLOREDEL D. GALON",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "RHEA C. GARCELLANO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ANGELO V. GARCIA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MERICK JAN U. NUEVACUBETA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "JANDI G. PANOLINO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ALVIN G. PALANCA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "SARAH JOANNE R. ROTIL",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "LEA M. CAMANGEG",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "JEFFREY H. DE CASTRO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "HERMENEGILDO P. DELA PEÑA",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MARY CLAIRE V. GATUCAO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MARIA ADELA J. LACAO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ABRILA M. LANGBAO",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "ANITA G. MALAZARTE",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "VERNALUZ C. MANGUSSAD",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "SWEET ANGELIKATE L. VILLARUEL",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MARIZEL L. YAYEN",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "MONICA JOYCE S. SESPEÑE",
    title: "FACULTY",
    
    photo: logoImage,
  },
  {
    name: "JAY - R P. VALDEZ",
    title: "FACULTY",
    
    photo: logoImage,
  },
];

const CollegeOrgan = () => {
  const [showChatbot, setShowChatbot] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
  const [showBubble, setShowBubble] = useState(true);
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const selectedFaculty = facultyMembers.slice(4, 6);


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
          >
            <Link to="/Services">Services</Link>
          </li>
          <li><Link to="/News">News</Link></li>
          <li><Link to="/ContactUs">Contact Us</Link></li>
        </ul>
      </div>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">COLLEGE ORGANIZATION GALLERY</h1>
        </div>
      </section>
            {/* docto */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[0].photo} alt={facultyMembers[0].name} className="faculty-photo" />
          <h3>{facultyMembers[0].name}</h3>
          <p><strong>{facultyMembers[0].title}</strong></p>
          <p>{facultyMembers[0].position}</p>
        </div>
      </section>
{/* vice */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[1].photo} alt={facultyMembers[1].name} className="faculty-photo" />
          <h3>{facultyMembers[1].name}</h3>
          <p><strong>{facultyMembers[1].title}</strong></p>
          <p>{facultyMembers[1].position}</p>
        </div>
      </section>

      {/* Dr. Ona */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[2].photo} alt={facultyMembers[2].name} className="faculty-photo" />
          <h3>{facultyMembers[2].name}</h3>
          <p><strong>{facultyMembers[2].title}</strong></p>
          <p>{facultyMembers[2].position}</p>
        </div>
      </section>

      {/* Mr. jandi */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[3].photo} alt={facultyMembers[3].name} className="faculty-photo" />
          <h3>{facultyMembers[3].name}</h3>
          <p><strong>{facultyMembers[3].title}</strong></p>
          <p>{facultyMembers[3].position}</p>
        </div>
      </section>
            {/* myrna and kalbo */}
      <div className="card-container">
        {selectedFaculty.map((member, index) => (
          <div key={index} className="card">
            <img src={member.photo} alt={member.name} className="card-photo" />
            <h3>{member.name}</h3>
            <p>{member.title}</p>
            <p>{member.position}</p>
          </div>
        ))}
      </div>

      <h1 className="title-card">Computer Studies Department</h1>
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[6].photo} alt={facultyMembers[6].name} className="faculty-photo" />
          <h3>{facultyMembers[6].name}</h3>
          <p><strong>{facultyMembers[6].title}</strong></p>
          <p>{facultyMembers[6].position}</p>
        </div>
      </section>

      {/* Remaining faculty in 4-column layout */}
      <section className="faculty-grid multi-column">
        {facultyMembers.slice(7).map((member, index) => (
          <div key={index} className="faculty-card">
            <img src={member.photo} alt={member.name} className="faculty-photo" />
            <h3>{member.name}</h3>
            <p><strong>{member.title}</strong></p>
            <p>{member.position}</p>
          </div>
        ))}
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

export default CollegeOrgan;
