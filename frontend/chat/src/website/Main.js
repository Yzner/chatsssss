
// import React, { useState, useEffect } from "react";
// import "../styles/Main.css"; 
// import ChatbotScreen from "../ChatbotScreen"; 
// import botIcon from "../chat.png"; 
// import { Link } from "react-router-dom"; 

// const Main = () => {
//   const [showChatbot, setShowChatbot] = useState(false);
//   const [scrolled, setScrolled] = useState(false);

//   const toggleChatbot = () => {
//     setShowChatbot(!showChatbot);
//   };

//   useEffect(() => {
//     const handleScroll = () => {
//       if (window.scrollY > 50) {
//         setScrolled(true);
//       } else {
//         setScrolled(false);
//       }
//     };

//     window.addEventListener("scroll", handleScroll);
//     return () => window.removeEventListener("scroll", handleScroll);
//   }, []);

//   return (
//     <div>
//       <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
//         <div className="logo">PalawanSu-CS</div>
//         <nav>
//           <ul>
//             <li><Link to="/">Home</Link></li>
//             <li><Link to="/about">About</Link></li> 
//             <li><a href="#organization">Organization</a></li>
//             <li><a href="#programs">Degree Programs</a></li>
//             <li><a href="#research">Research</a></li>
//             <li><a href="#resources">Student Resources</a></li>
//           </ul>
//         </nav>
//       </header>

//       <section className="welcome-section">
//         <div className="welcome-text">
//           <h1 className="college-title">College of Sciences</h1>
//           <p>Palawan State University</p>
//           <button className="know-more-btn">Know More</button>
//         </div>
//       </section>


//       <section className="news-events">
//         <h3>Welcome to the Official Website of the College of Sciences of Palawan State University.</h3>
//         <h2>News and Events</h2>
//         <div className="news-items">
//           <div className="news-item">
//             <img src="news1.jpg" alt="News 1" />
//             <p>News Title 1</p>
//           </div>
//           <div className="news-item">
//             <img src="news2.jpg" alt="News 2" />
//             <p>News Title 2</p>
//           </div>
//         </div>
//       </section>

//       <div className="chatbot-icon" onClick={toggleChatbot}>
//         <img src={botIcon} alt="Chatbot Icon" />
//       </div>

//       {showChatbot && <ChatbotScreen />}

//       <footer className="footer">
//         <p>© 2025 Palawan State University - College of Science</p>
//       </footer>
//     </div>
//   );
// };

// export default Main;




import React, { useState, useEffect } from "react";
import "../styles/Main.css";
import ChatbotScreen from "../ChatbotScreen";
import botIcon from "../chat.png";
import { Link } from "react-router-dom";

const Main = () => {
  const [showChatbot, setShowChatbot] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [showAboutDropdown, setShowAboutDropdown] = useState(false);
  const [showServicesDropdown, setShowServicesDropdown] = useState(false);


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
      </header>

      <section className="welcome-sec">
        <div className="welcome-text">
          <h1 className="college-title">College of Sciences</h1>
          <p>Palawan State University</p>
          <button className="know-more-btn">Know More</button>
        </div>
      </section>

      <section className="news-events">
        <h3>Welcome to the Official Website of the College of Sciences of Palawan State University.</h3>
        <h2>News and Events</h2>
        <div className="news-items">
          <div className="news-item">
            <img src="news1.jpg" alt="News 1" />
            <p>News Title 1</p>
          </div>
          <div className="news-item">
            <img src="news2.jpg" alt="News 2" />
            <p>News Title 2</p>
          </div>
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

export default Main;
