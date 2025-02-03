// import React, { useState, useEffect } from "react";
// import "../../styles/About.css"; 
// import { Link } from "react-router-dom"; 
// import ChatbotScreen from "../../ChatbotScreen";
// import botIcon from "../../chat.png";

// const StudentOrg = () => {
//   const [scrolled, setScrolled] = useState(false);
//   const [showChatbot, setShowChatbot] = useState(false);
//   const [showAboutDropdown, setShowAboutDropdown] = useState(false);
//     const [showServicesDropdown, setShowServicesDropdown] = useState(false);

//   const toggleChatbot = () => {
//     setShowChatbot(!showChatbot);
//   };

//   useEffect(() => {
//     const handleScroll = () => {
//       setScrolled(window.scrollY > 50);
//     };

//     window.addEventListener("scroll", handleScroll);
//     return () => window.removeEventListener("scroll", handleScroll);
//   }, []);

//   // Load Facebook SDK for the Page Plugin
//   useEffect(() => {
//     if (!window.FB) {
//       const script = document.createElement("script");
//       script.async = true;
//       script.defer = true;
//       script.crossOrigin = "anonymous";
//       script.src = "https://connect.facebook.net/en_US/sdk.js#xfbml=1&version=v12.0";
//       document.body.appendChild(script);
//     }
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
//           <h1 className="college-title">STUDENT ORGANIZATION</h1>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - College Student Government</h2>
//         <div className="Page-container">
//           <p>Follow us on Facebook:</p>

//           {/* Clickable Facebook Page Plugin */}
//           <a 
//             href="https://www.facebook.com/PSU.CS.CSG" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/PSU.CS.CSG" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - YBA: Young Biologist Association</h2>
//         <div className="Page-container">
//           <a 
//             href="https://www.facebook.com/psucsyba" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/psucsyba" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - ACS: Association of Computer Scientists </h2>
//         <div className="Page-container">
//           <a 
//             href="https://www.facebook.com/acs.psu" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/acs.psu" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - SITE: Society of Information Technology Enthusiasts </h2>
//         <div className="Page-container">
//           <a 
//             href="https://www.facebook.com/psu.site" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/psu.site" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - ESSA: Environmental Science Student Association </h2>
//         <div className="Page-container">
//           <a 
//             href="https://www.facebook.com/psu.cs.essa" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/psu.cs.essa" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
//         </div>
//       </section>

//       <section className="Page-section">
//         <h2>College of Sciences - MBS: Marine Biologists Society</h2>
//         <div className="Page-container">
//           <a 
//             href="https://www.facebook.com/psucsmbs" 
//             target="_blank" 
//             rel="noopener noreferrer"
//             style={{ display: "block", textDecoration: "none" }}
//           >
//             <div 
//               className="fb-page" 
//               data-href="https://www.facebook.com/psucsmbs" 
//               data-tabs="timeline" 
//               data-width="500" 
//               data-height="500" 
//               data-small-header="false" 
//               data-adapt-container-width="true" 
//               data-hide-cover="false" 
//               data-show-facepile="true"
//             ></div>
//           </a>
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

// export default StudentOrg;






import React, { useState, useEffect } from "react";
import "../../styles/About.css";
import { Link } from "react-router-dom";
import ChatbotScreen from "../../ChatbotScreen";
import botIcon from "../../chat.png";

const StudentOrg = () => {
  const [scrolled, setScrolled] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
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
      };
      document.body.appendChild(script);
    } else {
      window.FB.XFBML.parse();
    }
  }, []);
  

  const studentOrgs = [
    {
      name: "College Student Government",
      href: "https://www.facebook.com/PSU.CS.CSG",
    },
    {
      name: "Young Biologist Association",
      href: "https://www.facebook.com/psucsyba",
    },
    {
      name: "Association of Computer Scientists",
      href: "https://www.facebook.com/acs.psu",
    },
    {
      name: "Society of Information Technology Enthusiasts",
      href: "https://www.facebook.com/psu.site",
    },
    {
      name: "Environmental Science Student Association",
      href: "https://www.facebook.com/psu.cs.essa",
    },
    {
      name: "Marine Biologists Society",
      href: "https://www.facebook.com/psucsmbs",
    },
  ];

  return (
    <div>
      <header className={`navbar ${scrolled ? "scrolled" : ""}`}>
        <div className="logo">PalawanSU-CS</div>
        <nav>
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/about">About</Link></li>
            <li><Link to="/services">Services</Link></li>
            <li><a href="#news">News</a></li>
            <li><a href="#contact">Contact Us</a></li>
          </ul>
        </nav>
      </header>

      <section className="welcome-section">
        <div className="welcome-text">
          <h1 className="college-title">STUDENT ORGANIZATION</h1>
        </div>
      </section>

      <section className="Page-section">
        <div className="org-grid">
          {studentOrgs.map((org, index) => (
            <div className="org-card" key={index}>
              <h2>{org.name}</h2>
              <a href={org.href} target="_blank" rel="noopener noreferrer">
                <div 
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
              </a>
            </div>
          ))}
        </div>
      </section>

      <div className="chatbot-icon" onClick={() => setShowChatbot(!showChatbot)}>
        <img src={botIcon} alt="Chatbot Icon" />
      </div>

      {showChatbot && <ChatbotScreen />}

      <footer className="footer">
        <p>© 2025 Palawan State University - College of Science</p>
      </footer>
    </div>
  );
};

export default StudentOrg;
