// import React, { useState, useEffect } from "react";
// import "../../styles/About.css"; 
// import { Link } from "react-router-dom"; 
// import ChatbotScreen from "../../ChatbotScreen";
// import botIcon from "../../chat.png";

// const CollegeOrgan = () => {
//   const [scrolled, setScrolled] = useState(false);
//   const [showChatbot, setShowChatbot] = useState(false);
//   const [showAboutDropdown, setShowAboutDropdown] = useState(false);
//   const [showServicesDropdown, setShowServicesDropdown] = useState(false);

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
//         <div className="logo">PalawanSU-CS</div>
//         <nav>
//           <ul>
//             <li><Link to="/">Home</Link></li>
//             <li
//               className="dropdown"
//               onMouseEnter={() => setShowAboutDropdown(true)}
//               onMouseLeave={() => setShowAboutDropdown(false)}
//             >
//               <Link to="/about">About</Link>
//               {showAboutDropdown && (
//                 <ul className="dropdown-menu">
//                   <li><Link to="/MandV">University Mission & Vision</Link></li>
//                   <li><Link to="/GandO">College Goals and Objectives</Link></li>
//                   <li><Link to="/Programs">Academic Programs</Link></li>
//                   <li><Link to="/CollegeOrgan">Faculty & Staff</Link></li>
//                   <li><Link to="/StudentOrg">College Student Organizations</Link></li>
//                 </ul>
//               )}
//             </li>
//             <li
//                 className="dropdown"
//                 onMouseEnter={() => setShowServicesDropdown(true)}
//                 onMouseLeave={() => setShowServicesDropdown(false)}
//             >
//                 <Link to="/about">Services</Link>
//                 {showServicesDropdown && (
//                 <ul className="dropdown-menu">
//                     <li><Link to="/MandV">Academic Awards</Link></li>
//                     <li><Link to="/GandO">Procedures</Link></li>
//                     <li><Link to="/Programs">Enrollment</Link></li>
//                     <li><Link to="/CollegeOrgan">Email Request</Link></li>
//                 </ul>
//                 )}
//             </li>
//             <li><a href="#programs">News</a></li>
//             <li><a href="#research">Contact Us</a></li>
//           </ul>
//         </nav>
//       </header>

//       <section className="welcome-section">
//         <div className="welcome-text">
//           <h1 className="college-title">COLLEGE ORGANIZATION GALLERY</h1>
//         </div>
//       </section>

//       <section className="Name-section">
//         <h3>Dr. Ronald Edilberto A. Ona</h3>
//         <p>Dean, College of Sciences</p>
//         <p>Chair, College Bids & Awards Committee</p>
//       </section>

//       <section className="Name-section">
//         <h3>Dr. Ronald Edilberto A. Ona</h3>
//         <p>Dean, College of Sciences</p>
//         <p>Chair, College Bids & Awards Committee</p>
//       </section>

//       <section className="Name-section">
//         <h3>Dr. Ronald Edilberto A. Ona</h3>
//         <p>Dean, College of Sciences</p>
//         <p>Chair, College Bids & Awards Committee</p>
//       </section>

//       <section className="Name-section">
//         <h3>Dr. Ronald Edilberto A. Ona</h3>
//         <p>Dean, College of Sciences</p>
//         <p>Chair, College Bids & Awards Committee</p>
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

// export default CollegeOrgan;





import React, { useState, useEffect } from "react";
import "../../styles/About.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";
import deanPhoto from "../Pictures/ona.JPG"; 

const facultyMembers = Array(12).fill({
  name: "Dr. Ronald Edilberto A. Ona",
  title: "Dean, College of Sciences",
  position: "Chair, College Bids & Awards Committee",
  photo: deanPhoto,
});

const CollegeOrgan = () => {
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
          <h1 className="college-title">COLLEGE ORGANIZATION GALLERY</h1>
        </div>
      </section>

      <section className="faculty-grid">
        {facultyMembers.map((member, index) => (
          <div key={index} className="faculty-card">
            <img src={member.photo} alt={member.name} className="faculty-photo" />
            <h3>{member.name}</h3>
            <p>{member.title}</p>
            <p>{member.position}</p>
          </div>
        ))}
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

export default CollegeOrgan;
