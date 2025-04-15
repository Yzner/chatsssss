
import React, { useState, useEffect } from "react";
import "../../styles/Main.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  
import onaPhoto from "../Pictures/ona.JPG";
import royPhoto from "../Pictures/roy.jpg";
import renePhoto from "../Pictures/rene.JPG";
import kristinePhoto from "../Pictures/kristine.jpg";
import menchPhoto from "../Pictures/mench.JPG";
import ivanImage from "../Pictures/ivan.JPG";
import logoImage from "../Pictures/logocs.png";
import florPhoto from "../Pictures/flor.jpg";
import rizzaPhoto from "../Pictures/rizza.JPG";
import kimPhoto from "../Pictures/kim.JPG";
import bemPhoto from "../Pictures/bem.JPG";
import regPhoto from "../Pictures/reg.JPG";
import demyPhoto from "../Pictures/demy.JPG";
import divPhoto from "../Pictures/div.JPG";
import jenPhoto from "../Pictures/jen.JPG";
import maryPhoto from "../Pictures/mary.jpg";
import ericPhoto from "../Pictures/eric.JPG";
import adonImage from "../Pictures/adon.png";


const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const facultyMembers = [
  {
    name: "Dr. Ronald Edilberto A. Ona",
    title: "Dean, College of Sciences",
    position: "Chair, College Bids & Awards Committee",
    photo: onaPhoto,
  },
  {
    name: "Mr. Mike Jordan Mosquito",
    title: "MAT Biology/Education",
    position: "Associate Dean",
    photo: logoImage,
  },
  {
    name: "Mr. Rene V. Buliag",
    title: "Chairperson, Computer Studies Department",
    position: "Head, Research Department",
    photo: renePhoto,
  },
  {
    name: "Prof. Kristine Joy V. Martinez",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: kristinePhoto,
  },
  {
    name: "Ms. Menchie Lopez",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: menchPhoto,
  },
  {
    name: "Mr. Demy R. Dizon",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: demyPhoto,
  },
  {
    name: "Dr. Floredith Jeanne G. Alcid ",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: florPhoto,
  },
  {
    name: "Prof. Maria Rizza Laudiano-Armildez",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: rizzaPhoto,
  },
  {
    name: "Mr. Kim C. Baguio",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: kimPhoto,
  },
  {
    name: "Ms. Ma. Regina P. Bravo",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: regPhoto,
  },
  {
    name: "Prof. Bemsor T. Caabay",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: bemPhoto,
  },
  {
    name: "Ms. Divine P. Caabay",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: divPhoto,
  },
  {
    name: "Ms. Jennifer G. Rabang",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: jenPhoto,
  },
  {
    name: "Mr. Albert Ivan R. Castillo",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: ivanImage,
  },
  {
    name: "Mary Joy Delos Trinos",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: maryPhoto,
  },
  {
    name: "Ms. Tiffany O. Pabilona",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Ms. Ailene L. Sibayan-Bobier",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Mr. Eric Henry P. Rivera",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: ericPhoto,
  },
  {
    name: "Adonis C. Ampongan",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: adonImage,
  },
  {
    name: "Mr. Roi Cyril D. Dosado",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: royPhoto,
  },
  //MATH DEPARTMENT
  {
    name: "Dr. Marichu G. Mozo",
    title: "Chairperson, Mathematics Department",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Prof. Ma. Concepcion G. Abian",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Ms. Jessa Mae P. Abrina",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Prof. Juliana B. Arquero",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Ms. Blessy Ann L. Ayco",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Prof. Mary Jane A. Bundac",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Ms. Mary Joyce C. Dueñas",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "MR. Reuben Ysmael A. Ganapin",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Mr. Lelando H. Gaton",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Prof. Jerico M. Padrones",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Prof. Siote I. Wy",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  //BIO-PHYSICAL SCIENCE DEPARTMENT
  {
    name: "Imelda C. Pacaldo",
    title: "Chairperson, Bio-Physical Science Department",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Janeth S. Daganta",
    title: "Program Head- Medical Biology",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Renalyn o. Seguerra ",
    title: "Program Head- Marine Biology",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Rodolfo JR. O. Abalus",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Geronimo Allan Jerome G. Acosta",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Azareel Amon O. Alvarez",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Ryan N. Aranga",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Jaybie S. Arzaga",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Arselene U. Bitara",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Princess Eunice C. Denosta",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Jean Marie L. Diego",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Floredel D. Galon",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Rhea C. Garcellano",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Angelo V. Garcia",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Merick Jan U. Nuevacubeta",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Jandi G. Panolino",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Alvin G. Palanca",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Sarah Joanne R. Rotil",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Lea M. Camangeg",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Jeffrey H. De Castro",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Hermenegildo P. Dela Peña",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Mary Claire V. Gatucao",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Maria Adela J. Lacao",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Abrila M. Langbao",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Anita G. Malazarte",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Vernaluz C. Mangussad",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Sweet Angelikate L. Villaruel",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Marizel L. Yayen",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Monica Joyce S. Sespeñe",
    title: "Faculty Member",
    position: "Head, Research Department",
    photo: logoImage,
  },
  {
    name: "Jay - R P. Valdez",
    title: "Faculty Member",
    position: "Head, Research Department",
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

      {/* Dr. Ona */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[0].photo} alt={facultyMembers[0].name} className="faculty-photo" />
          <h3>{facultyMembers[0].name}</h3>
          <p><strong>{facultyMembers[0].title}</strong></p>
          <p>{facultyMembers[0].position}</p>
        </div>
      </section>

      {/* Mr. Mosquito */}
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[1].photo} alt={facultyMembers[1].name} className="faculty-photo" />
          <h3>{facultyMembers[1].name}</h3>
          <p><strong>{facultyMembers[1].title}</strong></p>
          <p>{facultyMembers[1].position}</p>
        </div>
      </section>

      <h1 className="title-card">Computer Studies Department</h1>
      <section className="faculty-grid single-column">
        <div className="faculty-card">
          <img src={facultyMembers[2].photo} alt={facultyMembers[2].name} className="faculty-photo" />
          <h3>{facultyMembers[2].name}</h3>
          <p><strong>{facultyMembers[2].title}</strong></p>
          <p>{facultyMembers[2].position}</p>
        </div>
      </section>

      {/* Remaining faculty in 4-column layout */}
      <section className="faculty-grid multi-column">
        {facultyMembers.slice(3).map((member, index) => (
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
