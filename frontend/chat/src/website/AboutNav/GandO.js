import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { FaChevronDown } from "react-icons/fa"; 
import "../../styles/Main.css"; 
import ChatbotScreen from "../../ChatbotScreen";
import botGif from "../Pictures/CHAT.gif";  
import logoImage from "../Pictures/logocs.png";
const chatbotMessages = [
  "Hi! You can ask me anything!",
  "Hi, I am Ask.CS!",
  "Ask me about PalawanSU College of Sciences!",
  "Welcome to the College of Sciences' Website."
];

const goalsData = [
  {
    id: "collegeGoals",
    title: "GOAL OF THE COLLEGE OF SCIENCES",
    description: "To advance the frontiers of knowledge in the environmental, computer, mathematical and biological sciences and to be a world-class center of learning, scientific inquiry and research beneficial to the general welfare of the People."
  },
  {
    id: "CScollegeGoals",
    title: "BS COMPUTER SCIENCE - PROGRAM GOALS",
    description: "To produce ethical professionals and researchers who are proficient in designing and developing computing solutions.",
    outcomes: [
      {
        section: "Program Outcomes - Computer Studies Department",
        list: [
          "Design an appropriate computing solution to solve complex problems.",
          "Apply computing and other knowledge domains to address real-world problems.",
          "Develop computing solutions using a system-level perspective.",
          "Utilize modern computing tools."
        ]
      },
      {
        section: "Program Outcomes - BS Computer Science",
        list: [
          "Recognize information security issues in relation to the design, development, and use of information systems.",
          "Design solutions for complex computing problems, systems, components, or processes that meet specific needs with appropriate consideration for public health and safety, culture, society, and environment.",
          "Function effectively as an individual and as a member or leader in diverse teams and in multidisciplinary settings.",
          "Communicate effectively about complex computing activities through writing effective reports, documentation, presentations and instructions.",
          "Apply professional, ethical and legal practices in computing technology.",
          "Engage in independent learning for continual development as a computing professional."
        ]
      }
    ]
  },
  {
    id: "EScollegeGoals",
    title: "BS ENVIRONMENTAL SCIENCE - PROGRAM GOALS",
    description: "To produce professionals equipped with relevant, updated and practical knowledge and skills in sustainably managing the environmental and natural resources as well as approximately responding to regional environmental issues and concerns.",
    outcomes: [
      {
        section: "Program Outcomes - BS Environmental Science",
        list: [
          "Demonstrate broad and coherent knowledge and understanding in the core areas of environmental science.",
          "Disseminate effectively knowledge pertaining to sound environmental protection, conservation, utilization and management.",
          "Demonstrate the ability to contribute to the protection and management of the environment.",
          "Analyze local environmental issues and problems in the regional and global context.",
          "Apply appropriate knowledge and innovation related to the environment."
        ]
      },
    ]
  },
  {
    id: "ITcollegeGoals",
    title: "BS INFORMATION TECHNOLOGY - PROGRAM GOALS",
    description: "To produce ethical IT professional who are well-versed on application, installation, operation, development, maintenance and administration of Information Technology Infrastructure. ",
    outcomes: [
      {
        section: "Program Outcomes - Computer Studies Department",
        list: [
          "Design an appropriate computing solution to solve complex problems.",
          "Apply computing and other knowledge domains to address real-world problems.",
          "Develop computing solutions using a system-level perspective.",
          "Utilize modern computing tools."
        ]
      },
      {
        section: "Program Outcomes - BS Information Technology",
        list: [
          "Apply computing standards.",
          "Analyze user needs and take them into account in the selection, creation, evaluation and administration of computer-based systems.",
          "Integrate IT-based solutions into the user environment effectively.",
          "Function effectively as a member or leader of a development team.",
          "Assist in the creation of an effective IT project plan.",
          "Communicate effectively about complex computing activities through logical writing, presentations, and clear instructions.",
          "Analyze the local and global impact of computing information technology on individuals, organizations, and society.",
          "Demonstrate understanding of professional, ethical, legal, security, and social issues and responsibilities in the utilization of information technology.",
          "Engage in self-learning and improving performance for continual professional development."
        ]
      }
    ]
  },
  {
    id: "YBAcollegeGoals",
    title: "BS MEDICAL BIOLOGY - PROGRAM GOALS",
    outcomes: [
      {
        section: "Program Outcomes - BS Medical Biology",
        list: [
          "Develop an in-depth understanding of the basic principles governing the science of life;",
          "Utilize techniques/procedures relevant to biological research work in laboratory or field settings;",
          "Apply basic mathematical and statistical computations and use of appropriate technologies in the analysis of biological data;",
          "Extend knowledge and critically assess current views and theories in various areas of the biological sciences",
        ]
      },
    ]
  },
  ,
  {
    id: "MBcollegeGoals",
    title: "BS MARINE BIOLOGY - PROGRAM GOALS",
    outcomes: [
      {
        section: "Program Outcomes - BS Medical Biology",
        list: [
          "Develop an in-depth understanding of the basic principles governing the science of life;",
          "Utilize techniques/procedures relevant to biological research work in laboratory or field settings;",
          "Apply basic mathematical and statistical computations and use of appropriate technologies in the analysis of biological data;",
          "Extend knowledge and critically assess current views and theories in various areas of the biological sciences",
        ]
      },
    ]
  },
];

const GoalsObjective = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  const [expandedSection, setExpandedSection] = useState(null);

  const toggleSection = (id) => setExpandedSection(expandedSection === id ? null : id);
  const [showBubble, setShowBubble] = useState(true); 
  const [chatbotMessage, setChatbotMessage] = useState(chatbotMessages[0]); 
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showAboutSideBar, setShowAboutSideBar] = useState(false);
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
    const handleScroll = () => setScrolled(window.scrollY > 50);
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
          <h1 className="college-title">GOALS AND OBJECTIVES</h1>
        </div>
      </section>

      {goalsData.map(({ id, title, description, outcomes }) => (
        <div key={id} className="expandable-section">
          <div className="goal-header" onClick={() => toggleSection(id)}>
            <h1>{title}</h1>
            <FaChevronDown className={`arrow-icon ${expandedSection === id ? "rotated" : ""}`} />
          </div>
          {expandedSection === id && (
            <div className="goal-content">
              <p className="goal-description">{description}</p>
              {outcomes && outcomes.map((outcome, index) => (
                <div key={index}>
                  <h3 className="outcome-section-title">{outcome.section}</h3>
                  <ol className="outcome-list">
                    {outcome.list.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ol>
                </div>
              ))}
            </div>
          )}
        </div>
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

export default GoalsObjective;
